#!/usr/bin/env python3
"""Export a quantized layer of a deployed YOLO model as PE command vectors.

Takes one Conv+BN block from a real checkpoint (default: v5.2 SAR ship
detector, layer 0), post-training-quantizes it TFLite-style with the PE's
Q15 arithmetic (model/pe_golden.py), runs real SAR chips through it, and
emits a JSON vector file: the LOAD_CFG block plus, per output pixel, the
exact (LOAD_W, MAC) byte sequence and the expected int8 EMIT result computed
by the golden model.

The same file drives three targets:
  - cocotb (test/test.py::test_real_layer_vectors) — RTL, today
  - the TT FPGA breakout via the demoboard MicroPython SDK — when purchased
  - the actual sky130 silicon — when it comes back from the shuttle

Quantization scheme (documented deviations from TFLite in brackets):
  - weights: per-tensor symmetric int8           [TFLite: per-channel]
  - activations in/out: asymmetric int8 with zero-point
  - BN folded into W', b' before quantization
  - bias + input-zero-point correction folded into SYNTHETIC MAC TERMS:
    the PE has no bias port, so the int32 correction C is decomposed into
    <= ~6 extra (w, a) pairs with |w*a| <= 127*127 that sum exactly to C.
  - activation: passthrough clamp (v5.2 uses SiLU, which the PE does not
    implement — vectors capture the pre-activation conv output). Pointing
    --model at the v5.4 Hard-Swish checkpoint and passing --hswish exports
    activation-matched vectors through the PE's hard-swish path instead.

Run with the training venv (torch) — pe_golden itself is pure Python:
  PYTHONPATH= /home/mark/code/aswarby/.venv-train/bin/python \
      tools/export_layer_vectors.py --n-pixels 16 \
      --out test/vectors/v5p2_layer0.json
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from pe_golden import GoldenPe, clamp, quantize_multiplier_q15  # noqa: E402

DEFAULT_MODEL = "/mnt/aswarby-data/training/checkpoints/v5/best-v5p2-deploy.pt"
DEFAULT_CHIPS = "/mnt/aswarby-data/training/unified/v5_sar/images/val"


def fold_bn(conv, bn):
    """Fold BatchNorm into conv weights/bias: W' = W*g/s, b' = beta + (b-mu)*g/s."""
    import torch

    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0])
    gamma, beta = bn.weight.detach(), bn.bias.detach()
    mu, var, eps = bn.running_mean.detach(), bn.running_var.detach(), bn.eps
    scale = gamma / (var + eps).sqrt()
    return w * scale.reshape(-1, 1, 1, 1), beta + (b - mu) * scale


def decompose_correction(c):
    """Split int32 correction C into (w, a) int8 pairs summing exactly to C."""
    pairs = []
    sign = 1 if c >= 0 else -1
    rem = abs(c)
    q, rem = divmod(rem, 127 * 127)
    pairs += [(sign * 127, 127)] * q
    q, rem = divmod(rem, 127)
    if q:
        pairs.append((sign * 127, q))
    if rem:
        pairs.append((sign * rem, 1))
    assert sum(w * a for w, a in pairs) == c
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--chips", default=DEFAULT_CHIPS)
    ap.add_argument("--n-chips", type=int, default=4)
    ap.add_argument("--n-pixels", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--out", default="test/vectors/v5p2_layer0.json")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    rng = random.Random(args.seed)

    # ---- load the layer ----------------------------------------------------
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = (ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt).float()
    block = model.model[args.layer]
    wf, bf = fold_bn(block.conv, block.bn)
    stride = block.conv.stride[0]
    pad = block.conv.padding[0]
    cout, cin, kh, kw = wf.shape
    print(f"layer {args.layer}: conv {cin}->{cout} k{kh} s{stride} p{pad}, BN folded")

    # ---- calibration: real SAR chips through the float layer ----------------
    files = sorted(f for f in os.listdir(args.chips) if f.lower().endswith((".jpg", ".png")))
    files = rng.sample(files, min(args.n_chips, len(files)))
    imgs = []
    for f in files:
        im = Image.open(os.path.join(args.chips, f)).convert("RGB").resize((320, 320))
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)  # NCHW in [0,1]
    with torch.no_grad():
        y = F.conv2d(x, wf, bf, stride=stride, padding=pad)    # pre-activation
    print(f"calibrated on {len(files)} chips; conv out range "
          f"[{y.min():.4f}, {y.max():.4f}]")

    # ---- quantization parameters --------------------------------------------
    s_in, zp_in = 1.0 / 255.0, -128                 # input [0,1] asymmetric
    s_w = float(wf.abs().max()) / 127.0             # weights symmetric per-tensor
    lo, hi = float(y.min()), float(y.max())
    s_out = (hi - lo) / 255.0
    zp_out = clamp(round(-128 - lo / s_out), -128, 127)
    m_real = s_in * s_w / s_out
    m0, n = quantize_multiplier_q15(m_real)
    w_q = torch.clamp(torch.round(wf / s_w), -127, 127).to(torch.int32)
    bias_q = torch.round(bf / (s_in * s_w)).to(torch.int64)
    print(f"s_w={s_w:.6g} s_out={s_out:.6g} zp_out={zp_out} "
          f"M={m_real:.6g} -> m0={m0} n={n}")

    # ---- golden PE with this config ------------------------------------------
    gold = GoldenPe()
    gold.m0, gold.n, gold.zp = m0, n, zp_out
    cfg_bytes = gold.cfg_bytes()

    # ---- sample output pixels and build op streams ----------------------------
    x_q = torch.clamp(torch.round(x / s_in) + zp_in, -128, 127).to(torch.int32)
    n_img, _, oh, ow = y.shape
    pixels = []
    diffs = []
    for _ in range(args.n_pixels):
        b = rng.randrange(n_img)
        c = rng.randrange(cout)
        # keep away from padding so the patch is fully real pixels
        oy = rng.randrange(1, oh - 1)
        ox = rng.randrange(1, ow - 1)
        iy, ix = oy * stride - pad, ox * stride - pad
        ops = []
        for ci in range(cin):
            for ky in range(kh):
                for kx in range(kw):
                    ops.append((int(w_q[c, ci, ky, kx]),
                                int(x_q[b, ci, iy + ky, ix + kx])))
        # bias + input-zp correction as synthetic MAC terms
        corr = int(bias_q[c]) - zp_in * int(w_q[c].sum())
        ops += decompose_correction(corr)

        gold.clear()
        for w, a in ops:
            gold.load_w(w)
            gold.mac(a)
        expected = gold.emit()

        y_ref = clamp(round(float(y[b, c, oy, ox]) / s_out) + zp_out, -128, 127)
        diffs.append(abs(expected - y_ref))
        pixels.append({"ops": ops, "expected": expected, "float_ref_q": y_ref})

    import statistics
    print(f"PE-vs-float-reference |diff|: max={max(diffs)} "
          f"mean={statistics.mean(diffs):.3f} (quantization error, not a bug)")

    out = {
        "meta": {
            "model": args.model, "layer": args.layer, "stride": stride,
            "conv": f"{cin}x{kh}x{kw}->{cout}", "chips": files,
            "s_in": s_in, "zp_in": zp_in, "s_w": s_w,
            "s_out": s_out, "zp_out": zp_out, "m0": m0, "n": n,
            "activation": "passthrough (pre-SiLU conv output)",
            "float_diff_max_lsb": max(diffs),
        },
        "cfg_bytes": cfg_bytes,
        "pixels": pixels,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: {len(pixels)} pixels, "
          f"{sum(len(p['ops']) for p in pixels)} MAC terms total")


if __name__ == "__main__":
    main()

"""P0 Hard-Swish error analysis (docs/PLAN.md).

Sweeps every int8 input at representative quantisation scales and compares the
fixed-point Hard-Swish (the GoldenPe scheme: u8 knees + Q15 multiplier) against
the float reference quantised back to int8. Writes docs/hardswish_error.png and
prints a per-scale summary table.

Run:  PYTHONPATH= .venv/bin/python model/hardswish_analysis.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pe_golden import clamp, hswish_params, rdiv_pot

# (scale, zero_point): first three have integral 3/s, 6/s (exact knees);
# the last two land mid-rounding — realistic worst cases.
CASES = [(3 / 127, 0), (0.05, -20), (0.1, 0), (0.043, 5), (0.0271, -64)]


def hswish_int(q, zp, p):
    u = q - zp
    v = clamp(u + p["q3"], 0, p["q6"])
    return clamp(rdiv_pot(u * v * p["mhs"], 15 + p["nhs"]) + zp, -128, 127)


def hswish_float_ref(q, scale, zp):
    x = scale * (q - zp)
    y = x * min(max(x + 3.0, 0.0), 6.0) / 6.0
    return clamp(round(y / scale) + zp, -128, 127)


def main():
    fig, (ax_curve, ax_err) = plt.subplots(1, 2, figsize=(11, 4.2))
    print(f"{'scale':>8} {'zp':>4} {'max|err|':>8} {'mean|err|':>9} {'exact%':>7}")
    summary = []
    for scale, zp in CASES:
        p = hswish_params(scale, zp)
        qs = range(-128, 128)
        got = [hswish_int(q, zp, p) for q in qs]
        ref = [hswish_float_ref(q, scale, zp) for q in qs]
        errs = [g - r for g, r in zip(got, ref)]
        mx = max(abs(e) for e in errs)
        mean = sum(abs(e) for e in errs) / len(errs)
        exact = 100.0 * sum(e == 0 for e in errs) / len(errs)
        print(f"{scale:>8.4f} {zp:>4} {mx:>8} {mean:>9.4f} {exact:>6.1f}%")
        summary.append((scale, zp, mx, exact))
        ax_err.plot(list(qs), errs, lw=0.9, label=f"s={scale:.4f}, zp={zp}")

    # representative transfer curve (one nice + one awkward scale)
    for scale, zp, style in [(0.05, -20, "-"), (0.043, 5, "--")]:
        p = hswish_params(scale, zp)
        qs = list(range(-128, 128))
        ax_curve.plot(qs, [hswish_float_ref(q, scale, zp) for q in qs],
                      "k" + style, lw=1, alpha=0.4,
                      label=f"float ref s={scale}")
        ax_curve.plot(qs, [hswish_int(q, zp, p) for q in qs],
                      style, lw=1, label=f"fixed-point s={scale}")
    ax_curve.set_xlabel("input q (int8)")
    ax_curve.set_ylabel("output q (int8)")
    ax_curve.set_title("Hard-Swish transfer: fixed-point vs float")
    ax_curve.legend(fontsize=7)
    ax_err.set_xlabel("input q (int8)")
    ax_err.set_ylabel("error (LSB)")
    ax_err.set_title("Fixed-point error, all int8 inputs")
    ax_err.set_yticks([-2, -1, 0, 1, 2])
    ax_err.legend(fontsize=7)
    fig.suptitle("tt_um_aswarby_pe P0 — Hard-Swish fixed-point scheme (u8 knees + Q15 mult)")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "hardswish_error.png")
    fig.savefig(out, dpi=140)
    print(f"\nwrote {os.path.abspath(out)}")
    worst = max(s[2] for s in summary)
    assert worst <= 2, f"scheme drifts beyond 2 LSB: {worst}"


if __name__ == "__main__":
    main()

# v2 Tiny Tapeout plan — `tt_um_aswarby_pe` (INT8 PE: MAC + requantize + activation)

**Created 2026-06-12. Target: SkyWater sky130 shuttle — opens July 2026, tapes out September (app deadline ~2026-09-07, confirm exact date + `tt-gds-action` tag when the shuttle opens in the TT app).**

Scope = Option A (full INT8 PE: v1 MAC core + requantize + ReLU) with Option B
(Hard-Swish/ReLU6 activation menu) as a gated bolt-on. Options C (weight register
file) and D (PE array) are explicitly OUT — they go to the FPGA board as v3
exploration. Plan assumes 1×2 tiles (€140) from day one; fitting 1×1 is a bonus,
never a constraint to design against.

---

## 1. Design specification

### 1.1 Datapath (what the chip computes)

Weight-stationary INT8 PE, byte-serial I/O, strobe-handshake (all inherited from v1):

```
acc      = saturating Σ (w_int8 × a_int8)            # v1 mac_core, unchanged math
requant  = clamp_int8( rnd(acc × M0 >> (15 + n)) + zp )   # NEW — Q15 scale
out      = activation(requant)                        # NEW — see 1.3
```

- **M0 in Q15 (int16), not TFLite's Q31 (int32).** This halves the requantize
  multiplier (32×16 instead of 32×32 → 64-bit product) at a relative precision
  cost of ~2⁻¹⁵, irrelevant for a demo PE. **Documented deviation** from the
  TFLite reference (Jacob et al. 2018) — state it in the README and quantify the
  error in the golden-model notebook. Fallback if area is desperate: Q12.
- **Rounding**: round-half-away-from-zero on the post-multiply shift (simpler
  than TFLite's doubling-high-mul + nudge; the golden model quantifies the
  difference; exhaustive small-domain tests pin the exact behaviour).
- **Output zero-point `zp`**: signed 9-bit add after the shift, then clamp to
  [-128, 127]. Signed-int8 output domain only (no uint8 mode).
- **Requantize is a separate command (`EMIT`)**, not fused after every MAC —
  accumulate K terms, then emit once. Matches real PE usage and keeps MAC
  throughput untouched.

### 1.2 Why ReLU/ReLU6 are nearly free

In the quantized output domain, fused ReLU/ReLU6 are just **clamp bounds**
(TFLite does exactly this): ReLU → clamp_min = zp; ReLU6 → clamp_max =
quantize(6.0). So the "activation menu" for A is two programmable clamp
registers — trivial area. **Hard-Swish is the only activation needing real
compute**, which is why it's the B bolt-on and the natural cut line.

### 1.3 Activation menu

| sel | fn         | implementation |
|-----|-----------|----------------|
| 00  | passthrough | clamp [-128,127] |
| 01  | ReLU       | clamp [zp, 127] |
| 10  | ReLU6      | clamp [zp, q(6)] |
| 11  | Hard-Swish (B, gated) | int16 intermediate: h = clamp(x+q(3), 0, q(6)); y = (x·h·C₁₆) >> s where C₁₆ ≈ ⌈2¹⁶/6·scale⌉ |

Hard-Swish costs one int8×int9 multiply plus one constant multiply (or shift-add
tree) — pipeline it over 2–3 stages. Exact fixed-point scheme is **decided by the
golden model in P0**, not in RTL.

### 1.4 Interface / pin map (delta from v1)

Keep the v1 strobe handshake verbatim (one strobe edge = one op) so the cocotb
harness ports over. Changes:

- **cmd widens 2→3 bits** (`uio_in[2:0]`; strobe moves to `uio_in[3]`, rd_sel to
  `uio_in[5:4]`): NOP / LOAD_W / MAC / CLEAR / LOAD_CFG / EMIT / (2 spare).
- **LOAD_CFG** writes an auto-incrementing config pointer (M0.lo, M0.hi,
  shift n + act sel, zp, [hard-swish consts]) — one opcode, no cmd-space
  pressure if config grows.
- **uo_out** carries the int8 result directly after EMIT; rd_sel still selects
  acc bytes for debug readback (keep — it earned its keep in v1 bring-up).
- done on `uio_out[6]`, sticky ovf on `uio_out[7]` (or keep v1 positions if the
  remap annoys — decide at RTL time, document in info.yaml).

### 1.5 Pipeline (the v1 lesson, applied from day one)

v1 retrofitted a 4-stage pipeline after a −27 ns WNS failure; the worst path was
the bare 8×8 multiply. v2's 32×16 requantize multiply is bigger — **design it
split-and-pipelined from the start**:

- Requant: 2×(16×16) half-products → reconstruct → round+shift → zp+clamp
  ≈ 4–5 stages.
- Hard-Swish: +2–3 stages.
- Total EMIT latency ~8–10 cycles behind a DONE_DELAY shift register exactly like
  v1. Throughput is irrelevant (strobe-spaced ops); latency is cheap; slack is
  what we buy. sky130 is faster than GF180, so closing 50 MHz typical with this
  depth should be comfortable — but verify at the first harden, never assume.

---

## 2. Verification plan

- **P0 golden model first** (`model/pe_golden.py`, numpy): MAC (port v1's),
  requantize with the exact Q15/rounding scheme, all four activations.
  **Cross-check ReLU/ReLU6 paths against the real TFLite/LiteRT quantized
  kernels** on small cases — that cross-check is the credibility anchor and
  decides the Hard-Swish fixed-point scheme empirically (error histogram vs
  float reference in a small notebook).
- **cocotb suite** (port v1 patterns, Icarus + cocotb 2.0.1 from the python3.8
  user site — same toolchain quirk as v1, PATH note applies):
  - requant rounding edges: ±half-LSB, negative values, acc = INT32_MIN/MAX,
    M0 extremes, shift 0 and max;
  - zp extremes, clamp boundaries per activation;
  - randomized sweeps vs golden (thousands of cases);
  - **end-to-end real-layer test**: extend v1's `tools/export_vectors.py` to dump
    weights + M0 + shift + zp from a real quantized layer of a deployed model
    (v5.2 Hailo-quantized YOLO is the obvious source) and verify the PE
    reproduces the layer's int8 outputs term-for-term. This is also the demo.
  - Gate-level: exhaustive sweeps auto-skip under GATES=yes (v1 pattern).
- **FPGA hardware replay** (P4): same vectors via demoboard MicroPython SDK.
  Function-only (UP5K maps multiplies to SB_MAC16 DSPs — proves RTL + SDK +
  pinout, NOT ASIC timing).

---

## 3. Phases (each independently verifiable)

| Phase | Deliverable | Verify by | Gate |
|-------|------------|-----------|------|
| **P0** Spec + golden model | `pe_golden.py` + error notebook; Q15 + rounding + Hard-Swish scheme frozen | golden vs LiteRT kernels on ReLU/ReLU6 cases; Hard-Swish error histogram acceptable | — |
| **P1** Scaffold + v1 port | New repo `tt-aswarby-pe` from `ttsky-verilog-template` (sky130A); v1 mac_core/mac_fsm ported; cmd field widened; cocotb green | 7/7 v1 tests pass on sky130 sim; **first harden of skeleton** | **G1: area/timing baseline** — % of 1×2, WNS |
| **P2** Requantize | Pipelined 32×16 requant + LOAD_CFG + EMIT; full cocotb requant suite green | randomized sweeps vs golden; harden | **G2: fits with slack?** sets B's budget |
| **P3** Activations | ReLU/ReLU6 clamp bounds (cheap, always in). Then **B decision**: margin at G2 → implement Hard-Swish, re-harden | activation tests vs golden; real-layer vector test passes | **G3: B in or out** (cut line = Hard-Swish) |
| **P4** FPGA bring-up | Bitstream via `tt_fpga.py harden`; demoboard SDK replay of real-layer vectors on hardware | vectors pass on UP5K | hardware-verified RTL |
| **P5** Submission | All four TT gates green (gds/precheck/gl_test/viewer), docs + info.yaml + 3D viewer Pages, submit | TT app shows accepted | **submitted well before Sept 7** |

P4 can run in parallel with P5 prep; P4 is not a blocker for submission if the
kit ships late (cocotb + GL sim remain the formal verification basis — v1
precedent).

### Timeline sketch (runway is generous; phases coexist with model-training arcs)

- **June**: P0–P2. Order hardware in week 1.
- **Early July**: P3, P4. Shuttle opens → confirm deadline, tile pricing, action tag.
- **Mid/late July**: P5 submit. ~6 weeks of slack to the September close.

---

## 4. Immediate actions (week 1)

1. **[user] Order the FPGA Development Kit** (€90, store.tinytapeout.com — 9 in
   stock on 2026-06-12, preliminary-firmware note applies) **and check/order a
   TT demoboard** (sold separately; needed for v1 GF26B bring-up anyway; current
   FPGA kit revision is NOT compatible with TT08-and-earlier demoboards).
2. **[claude] P0**: write `pe_golden.py` + the LiteRT cross-check notebook;
   freeze the arithmetic spec.
3. **[claude] P1**: fork/scaffold `tt-aswarby-pe` (gmail git identity; public on
   github.com/markd666), port v1 core, first harden for G1.

## 5. Risks

| Risk | Mitigation |
|------|-----------|
| 32×16 requant multiply blows area | 1×2 budgeted from day one; Q15→Q12 fallback; Hard-Swish is the cut line |
| Rounding-mode bugs (classic quantization trap) | golden model frozen in P0 before RTL; exhaustive small-domain sweeps; LiteRT cross-check |
| sky130 timing surprise despite faster process | pipeline designed-in; G1 baseline harden before any new datapath lands |
| Shuttle details unknown until July | nothing blocks on them; confirm tag/deadline at open; submit margin ~6 weeks |
| FPGA kit stock/lead time | order week 1; P4 explicitly non-blocking for submission |
| Pin remap breaks v1 harness reuse | handshake semantics unchanged; only field positions move; single constants file shared by RTL + cocotb |

## 6. Out of scope (parked for v3)

Weight register file / micro-neuron (Option C) and multi-PE array (Option D) —
prototype on the UP5K (5.3k LUTs, 128 KB SPRAM hosts both trivially) once P4
infrastructure exists. Standalone requantizer (E) and sky130 port of bare v1 (F)
are dead unless G2 catastrophically fails; F's cross-process datapoint falls out
of G1/G2 harden reports for free anyway.

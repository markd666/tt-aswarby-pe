# INT8 PE: MAC + requantize + activation

## How it works

A byte-serial, weight-stationary **INT8 processing element** — the compute
primitive of a quantized neural-network accelerator, and the successor to the
`tt-aswarby-mac` GF180 design (GF26B shuttle). v2 extends the proven MAC core
with the full post-accumulation stage of real INT8 inference: fixed-point
requantization and fused activations (ReLU / ReLU6 as programmable clamp
bounds, Hard-Swish as a dedicated fixed-point datapath).

The datapath follows the TFLite integer-inference scheme with one documented
deviation: the requantize scale multiplier is **Q15** (16-bit mantissa) rather
than TFLite's Q31, because the int8 output clamp makes the extra mantissa
precision unobservable (cross-checked bit-level against a verbatim port of the
gemmlowp reference kernels: >=97.9% exact, never more than 1 LSB apart).

Commands are issued byte-serially over a strobe handshake (one strobe edge =
one operation), 3-bit command on `uio[2:0]`:

| cmd | operation |
|-----|-----------|
| 000 | NOP |
| 001 | LOAD_W — load the stationary INT8 weight from `ui_in` |
| 010 | MAC — acc += weight × activation (`ui_in`), saturating INT32 |
| 011 | CLEAR — zero the accumulator + sticky overflow flag |
| 100 | LOAD_CFG — write config registers (auto-incrementing pointer) |
| 101 | EMIT — requantize the accumulator to INT8 + apply activation |

The MAC multiply is split into nibble half-products over a 4-stage pipeline
(carried over from the GF180 design, where a single-cycle 8×8 multiply could
not close timing); `done` (`uio[6]`) pulses when each operation commits.
`uio[7]` is a sticky saturation flag. The accumulator is readable
byte-by-byte via `uio[5:4]` for debug.

The arithmetic contract is frozen in `model/pe_golden.py` (pure-Python golden
model, the single source of truth for both the cocotb testbench and the
host-side configuration helpers).

## How to test

Run the cocotb testbench in `test/` (`make -C test`) — it drives the full
command set against the golden model, including exhaustive INT32 saturation
sweeps in both directions and randomized MAC/requantize sequences.

On the demoboard: enable the project, then strobe commands per the table
above. Quick smoke test: LOAD_W 3, MAC 5, then read the accumulator back
LSB-first with `rd_sel` 0→3 — expect 15. A full real-layer demo (weights and
quantization parameters exported from a deployed INT8 YOLO model) lives in
the repo's vector-replay tooling.

## External hardware

None — fully digital, drivable from the demoboard's RP2040 MicroPython SDK.

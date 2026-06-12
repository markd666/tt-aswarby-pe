"""Bit-accurate golden model for tt_um_aswarby_pe (v2 INT8 PE).

This file IS the frozen arithmetic spec (P0 of docs/PLAN.md); the RTL is
written to match it, never the other way round. Pure-int Python with no
third-party imports so the cocotb testbench can import it directly, in the
same style as v1's GoldenMac.

Datapath:
    acc  = saturating INT32 sum of (w_int8 * a_int8)        # v1 MAC, unchanged
    r    = round_half_away( acc * M0  >>  (15 + n) )        # Q15 requantize
    q    = clamp( r + ZP, -128, 127 )                       # linear int8 point
    out  = clamp(q, QMIN, QMAX)            if ACT_CLAMP     # ReLU/ReLU6 = bounds
         = hard_swish(q)                   if ACT_HSWISH    # see emit()

Q15-not-Q31 justification (the documented TFLite deviation): the mantissa
truncation has relative error <= 2**-16, and the only values that survive the
final clamp are |r| <~ 128, where that relative error is an absolute error of
~0.002 LSB.  Q31 buys nothing once the output domain is int8.

Config register file (LOAD_CFG auto-increment order — shared with RTL):
    0: M0[7:0]      1: M0[15:8]     2: act_sel<<5 | n[4:0]
    3: ZP           4: QMIN         5: QMAX
    6: Q3 (u8)      7: Q6 (u8)      8: MHS[7:0]
    9: MHS[15:8]   10: NHS[4:0]
"""

INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)

ACT_CLAMP = 0
ACT_HSWISH = 1


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def rdiv_pot(x, exp):
    """Rounding right shift: nearest, ties away from zero. exp >= 0."""
    if exp == 0:
        return x
    half = 1 << (exp - 1)
    if x >= 0:
        return (x + half) >> exp
    return -((-x + half) >> exp)


def quantize_multiplier_q15(real):
    """Decompose a real multiplier into (m0, n): real ~= m0 / 2**15 * 2**-n,
    with m0 normalised to [2**14, 2**15).  Host-side helper (mirrors TFLite's
    QuantizeMultiplier, but Q15 and right-shift-only: requires 0 < real < 1)."""
    if not 0.0 < real < 1.0:
        raise ValueError(f"multiplier must be in (0, 1), got {real}")
    n = 0
    while real < 0.5:
        real *= 2.0
        n += 1
        if n > 31:
            raise ValueError("multiplier too small for 5-bit shift field")
    m0 = round(real * (1 << 15))
    if m0 == 1 << 15:  # rounding nudge, same trick as TFLite
        m0 >>= 1
        n -= 1
        if n < 0:
            raise ValueError("multiplier rounds to 1.0; fold into the layer")
    assert (1 << 14) <= m0 < (1 << 15)
    return m0, n


def hswish_params(scale, zero_point):
    """Host-side helper: hard-swish config for a tensor quantised as
    value = scale * (q - zero_point), shared input/output quant params.
    Constraint: scale >= 3/255 (~0.0118) so Q3/Q6 fit in u8."""
    q3 = round(3.0 / scale)
    q6 = round(6.0 / scale)
    if q6 > 255:
        raise ValueError(f"scale {scale} too small: Q6={q6} exceeds u8")
    mhs, nhs = quantize_multiplier_q15(scale / 6.0)
    return dict(q3=q3, q6=q6, mhs=mhs, nhs=nhs)


class GoldenPe:
    """Weight-stationary INT8 PE: v1 GoldenMac semantics + requantize/activation."""

    def __init__(self):
        self.weight = 0
        self.acc = 0
        self.ovf = 0
        # config registers (reset values)
        self.m0 = 1 << 14   # 0.5 in Q15
        self.n = 0
        self.zp = 0
        self.qmin = -128
        self.qmax = 127
        self.act = ACT_CLAMP
        self.q3 = 0
        self.q6 = 0
        self.mhs = 1 << 14
        self.nhs = 0

    # ---- v1 MAC core, verbatim semantics ------------------------------
    def load_w(self, data_s8):
        self.weight = data_s8

    def mac(self, data_s8):
        self.acc += data_s8 * self.weight
        if self.acc > INT32_MAX:
            self.acc = INT32_MAX
            self.ovf = 1
        elif self.acc < INT32_MIN:
            self.acc = INT32_MIN
            self.ovf = 1

    def clear(self):
        self.acc = 0
        self.ovf = 0

    # ---- v2 requantize + activation ------------------------------------
    def requant_linear(self):
        """acc -> int8 linear point (pre-activation), saturating."""
        t = self.acc * self.m0          # int32 * u16 -> fits in 48 bits
        r = rdiv_pot(t, 15 + self.n)
        return clamp(r + self.zp, -128, 127)

    def emit(self):
        q = self.requant_linear()
        if self.act == ACT_CLAMP:
            return clamp(q, self.qmin, self.qmax)
        # hard-swish: y = x * relu6(x + 3) / 6, all in the tensor's int domain
        u = q - self.zp                       # int9   [-255, 255]
        v = clamp(u + self.q3, 0, self.q6)    # u8     [0, 255]
        p = u * v                             # int17
        y = rdiv_pot(p * self.mhs, 15 + self.nhs) + self.zp
        return clamp(y, -128, 127)

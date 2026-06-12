"""P0 verification of the golden model (docs/PLAN.md).

Three jobs:
  1. pin the rounding/saturation primitives with exact-value tests;
  2. cross-check the Q15 requantize against the verbatim TFLite/gemmlowp
     reference (tflite_reference.py) — the credibility anchor;
  3. bound the Hard-Swish fixed-point error against a float reference.
"""

import math
import random

import pytest

from pe_golden import (
    ACT_HSWISH,
    GoldenPe,
    INT32_MAX,
    INT32_MIN,
    clamp,
    hswish_params,
    quantize_multiplier_q15,
    rdiv_pot,
)
from tflite_reference import (
    quantize_multiplier_q31,
    requantize_reference,
    rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
)

# Every randomized test seeds its own stream so tests stay order-independent
# (a module-level seed makes pass/fail depend on which tests run before you).


def _rand_real_multiplier(rng):
    """A scale inside quantize_multiplier_q15's documented domain: the helper
    rejects real >= 1 - 2**-16 (rounds to 1.0 in Q15), so cap below that."""
    return 10 ** rng.uniform(-5, -1e-4)  # (1e-5, ~0.99977)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "x,exp,expected",
    [
        (0, 4, 0),
        (8, 4, 1),      # +0.5 -> 1   (ties away from zero)
        (-8, 4, -1),    # -0.5 -> -1
        (7, 4, 0),      # just under half
        (-7, 4, 0),
        (9, 4, 1),
        (-9, 4, -1),
        (24, 4, 2),     # +1.5 -> 2
        (-24, 4, -2),
        (5, 0, 5),      # exp=0 passthrough
        (-1, 1, -1),    # -0.5 -> -1 again, minimal case
    ],
)
def test_rdiv_pot_half_away(x, exp, expected):
    assert rdiv_pot(x, exp) == expected


def test_srdhm_known_vectors():
    # the documented single-overflow case
    assert saturating_rounding_doubling_high_mul(INT32_MIN, INT32_MIN) == INT32_MAX
    # exact halves: a*b = 0.5 * 2**31 and -0.5 * 2**31
    assert saturating_rounding_doubling_high_mul(1 << 15, 1 << 15) == 1   # +0.5 -> 1
    assert saturating_rounding_doubling_high_mul(-(1 << 15), 1 << 15) == 0  # -0.5 -> 0 (gemmlowp trunc)
    assert saturating_rounding_doubling_high_mul(0, INT32_MAX) == 0


def test_srdhm_matches_real_arithmetic():
    rng = random.Random(1)
    for _ in range(20000):
        a = rng.randint(INT32_MIN, INT32_MAX)
        b = rng.randint(INT32_MIN, INT32_MAX)
        got = saturating_rounding_doubling_high_mul(a, b)
        assert abs(got - a * b / 2**31) <= 1.0


def test_rounding_divide_by_pot_half_away():
    assert rounding_divide_by_pot(8, 4) == 1
    assert rounding_divide_by_pot(-8, 4) == -1
    assert rounding_divide_by_pot(7, 4) == 0
    assert rounding_divide_by_pot(-7, 4) == 0


# ---------------------------------------------------------------------------
# Multiplier decomposition helpers
# ---------------------------------------------------------------------------
def test_quantize_multiplier_q15_roundtrip():
    rng = random.Random(2)
    for _ in range(5000):
        real = _rand_real_multiplier(rng)
        m0, n = quantize_multiplier_q15(real)
        assert (1 << 14) <= m0 < (1 << 15)
        recon = m0 / (1 << 15) * 2.0**-n
        assert math.isclose(recon, real, rel_tol=2**-15)


def test_quantize_multiplier_q15_rejects_out_of_range():
    with pytest.raises(ValueError):
        quantize_multiplier_q15(1.0)
    with pytest.raises(ValueError):
        quantize_multiplier_q15(0.0)
    with pytest.raises(ValueError):
        quantize_multiplier_q15(1e-12)  # needs shift > 31


# ---------------------------------------------------------------------------
# MAC core (v1 semantics, ported verbatim)
# ---------------------------------------------------------------------------
def test_mac_saturation_both_directions_and_sticky_ovf():
    pe = GoldenPe()
    pe.load_w(127)
    pe.acc = INT32_MAX - 100
    pe.mac(127)  # +16129 pushes past the rail
    assert pe.acc == INT32_MAX and pe.ovf == 1
    pe.clear()
    assert pe.acc == 0 and pe.ovf == 0
    pe.load_w(-128)
    pe.acc = INT32_MIN + 100
    pe.mac(127)  # -16256
    assert pe.acc == INT32_MIN and pe.ovf == 1


def test_mac_accumulates_exactly():
    pe = GoldenPe()
    rng = random.Random(3)
    total = 0
    for _ in range(1000):
        w = rng.randint(-128, 127)
        a = rng.randint(-128, 127)
        pe.load_w(w)
        pe.mac(a)
        total += w * a
    assert pe.acc == total and pe.ovf == 0


# ---------------------------------------------------------------------------
# Q15 requantize vs the TFLite Q31 reference (the P0 cross-check)
# ---------------------------------------------------------------------------
def _configured_pe(real, zp):
    pe = GoldenPe()
    pe.m0, pe.n = quantize_multiplier_q15(real)
    pe.zp = zp
    return pe


def test_requant_matches_tflite_within_1lsb():
    """Accs targeted so the result lands in/near the int8 range — the only
    region where Q15-vs-Q31 can differ post-clamp (see pe_golden docstring)."""
    rng = random.Random(4)
    exact = 0
    total = 0
    for _ in range(20000):
        real = _rand_real_multiplier(rng)
        zp = rng.randint(-128, 127)
        r_target = rng.uniform(-160, 160)
        acc = clamp(round(r_target / real), INT32_MIN, INT32_MAX)
        pe = _configured_pe(real, zp)
        pe.acc = acc
        ours = pe.emit()
        ref = requantize_reference(acc, real, zp)
        assert abs(ours - ref) <= 1, (acc, real, zp, ours, ref)
        exact += ours == ref
        total += 1
    rate = exact / total
    print(f"\nQ15 vs Q31 exact-match rate (in-range accs): {rate:.4%}")
    assert rate > 0.95


def test_requant_extreme_accs_saturate_identically():
    rng = random.Random(5)
    for _ in range(5000):
        real = _rand_real_multiplier(rng)
        zp = rng.randint(-128, 127)
        acc = rng.choice(
            [rng.randint(INT32_MIN, INT32_MAX), INT32_MIN, INT32_MAX, 0]
        )
        pe = _configured_pe(real, zp)
        pe.acc = acc
        assert abs(pe.emit() - requantize_reference(acc, real, zp)) <= 1


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
def test_relu_and_relu6_are_clamp_bounds():
    real, zp = 0.013, -10
    pe = _configured_pe(real, zp)
    pe.qmin = zp  # fused ReLU
    for acc in (-(10**6), -1, 0, 1, 10**4, 10**7):
        pe.acc = acc
        out = pe.emit()
        assert out >= zp  # quantised zero
        if acc <= 0:
            assert out == zp
    q6 = zp + round(6.0 / real)
    pe.qmax = min(127, q6)  # fused ReLU6
    pe.acc = 10**9
    assert pe.emit() == min(127, q6)


def _hswish_float_ref(q, scale, zp):
    x = scale * (q - zp)
    y = x * min(max(x + 3.0, 0.0), 6.0) / 6.0
    return clamp(round(y / scale) + zp, -128, 127)


@pytest.mark.parametrize("scale,zp", [(3 / 127, 0), (0.05, -20), (0.1, 0)])
def test_hardswish_within_1lsb_nice_scales(scale, zp):
    """Scales where 3/s and 6/s are integral (no knee-rounding error)."""
    pe = GoldenPe()
    pe.act = ACT_HSWISH
    pe.zp = zp
    for k, v in hswish_params(scale, zp).items():
        setattr(pe, k, v)
    worst = 0
    for q in range(-128, 128):
        # drive emit() through requant_linear by planting acc so q comes out:
        # simplest is to bypass — set m0/n/zp for identity-ish, or monkey the
        # linear point directly via acc = (q - zp) << (15 + n) / m0. Cleaner:
        # test the activation stage in isolation.
        u = q - pe.zp
        v_ = clamp(u + pe.q3, 0, pe.q6)
        y = rdiv_pot(u * v_ * pe.mhs, 15 + pe.nhs) + pe.zp
        got = clamp(y, -128, 127)
        worst = max(worst, abs(got - _hswish_float_ref(q, scale, zp)))
    assert worst <= 1, f"max hard-swish error {worst} LSB at scale={scale}"


def test_hardswish_non_nice_scale_within_2lsb():
    scale, zp = 0.043, 5  # 3/s, 6/s land mid-rounding: worst realistic case
    pe = GoldenPe()
    pe.act = ACT_HSWISH
    pe.zp = zp
    for k, v in hswish_params(scale, zp).items():
        setattr(pe, k, v)
    worst = 0
    for q in range(-128, 128):
        u = q - zp
        v_ = clamp(u + pe.q3, 0, pe.q6)
        y = clamp(rdiv_pot(u * v_ * pe.mhs, 15 + pe.nhs) + zp, -128, 127)
        worst = max(worst, abs(y - _hswish_float_ref(q, scale, zp)))
    print(f"\nhard-swish worst error at awkward scale {scale}: {worst} LSB")
    assert worst <= 2


def test_hswish_params_rejects_tiny_scale():
    with pytest.raises(ValueError):
        hswish_params(0.01, 0)  # Q6 = 600 > u8

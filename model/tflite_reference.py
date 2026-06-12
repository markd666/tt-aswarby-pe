"""Verbatim Python port of the TFLite/gemmlowp reference requantize arithmetic.

This is the cross-check oracle for pe_golden's Q15 scheme (P0 of docs/PLAN.md).
Ported function-for-function from gemmlowp fixedpoint.h / TFLite
QuantizeMultiplier so the comparison is against the *actual* kernel math, not
a paraphrase. C++ int semantics (truncating division, arithmetic shifts) are
reproduced explicitly.
"""

import math

INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)


def _trunc_div(a, b):
    """C++ integer division: truncates toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def saturating_rounding_doubling_high_mul(a, b):
    """gemmlowp SaturatingRoundingDoublingHighMul(int32, int32) -> int32."""
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX  # the single overflow case
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    return _trunc_div(ab + nudge, 1 << 31)


def rounding_divide_by_pot(x, exponent):
    """gemmlowp RoundingDivideByPOT: nearest, ties away from zero."""
    assert 0 <= exponent <= 31
    mask = (1 << exponent) - 1
    remainder = x & mask                      # same low-bit semantics as C++
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def multiply_by_quantized_multiplier(acc, quantized_multiplier, shift):
    """TFLite MultiplyByQuantizedMultiplier. shift <= 0 here (multiplier < 1);
    the left-shift branch exists in TFLite but is unreachable for requantize
    scales, matching the hardware's right-shift-only field."""
    assert shift <= 0
    return rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(acc, quantized_multiplier), -shift
    )


def quantize_multiplier_q31(real):
    """TFLite QuantizeMultiplier: real -> (q31 mantissa, shift), mantissa in
    [2**30, 2**31), real ~= mantissa / 2**31 * 2**shift."""
    if real == 0.0:
        return 0, 0
    m, e = math.frexp(real)          # real = m * 2**e, m in [0.5, 1)
    q = round(m * (1 << 31))
    if q == 1 << 31:
        q //= 2
        e += 1
    return q, e


def requantize_reference(acc, real_multiplier, zero_point):
    """Full TFLite-style requantize of an int32 accumulator to int8."""
    q31, shift = quantize_multiplier_q31(real_multiplier)
    r = multiply_by_quantized_multiplier(acc, q31, shift)
    return max(-128, min(127, r + zero_point))

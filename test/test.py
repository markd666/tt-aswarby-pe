# SPDX-FileCopyrightText: © 2026 Mark Shilton
# SPDX-License-Identifier: Apache-2.0
"""cocotb testbench for tt_um_aswarby_pe (P1: v1 MAC behind the v2 interface).

The golden model is imported from model/pe_golden.py — the frozen P0 spec —
rather than redefined here (the v1 testbench inlined its GoldenMac; v2 keeps
exactly one source of arithmetic truth).

v2 pin map under test:
    uio_in[2:0] cmd / uio_in[3] strobe / uio_in[5:4] rd_sel
    uio_out[6] done / uio_out[7] ovf
"""

import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from pe_golden import (  # noqa: E402
    ACT_CLAMP,
    ACT_HSWISH,
    INT32_MAX,
    INT32_MIN,
    GoldenPe,
    hswish_params,
    quantize_multiplier_q15,
)

# Exhaustive saturation sweeps are minutes-long at RTL and unusable at gate
# level, so they always skip under gate-level sim (GATES=yes) and can be
# skipped on demand with SKIP_SLOW=1.  (v1 convention, unchanged.)
_SKIP_SLOW = os.environ.get("SKIP_SLOW") == "1" or os.environ.get("GATES") == "yes"

# v2 command encoding (3 bits; 000..011 match v1's 2-bit codes)
CMD_NOP = 0
CMD_LOADW = 1
CMD_MAC = 2
CMD_CLEAR = 3
CMD_LOAD_CFG = 4  # inert until P2
CMD_EMIT = 5      # inert until P2


# ----------------------------------------------------------------------------
# Pin helpers
# ----------------------------------------------------------------------------
def s8_bits(v):
    """Two's-complement byte for a signed value in [-128, 127]."""
    return v & 0xFF


def uio_inputs(cmd=0, strobe=0, rd_sel=0):
    return (cmd & 7) | ((strobe & 1) << 3) | ((rd_sel & 3) << 4)


def done_flag(dut):
    return (int(dut.uio_out.value) >> 6) & 1


def ovf_flag(dut):
    return (int(dut.uio_out.value) >> 7) & 1


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def op(dut, cmd, data=0):
    """Present cmd+data, pulse strobe once, wait for the done flag, re-arm."""
    dut.ui_in.value = s8_bits(data)
    dut.uio_in.value = uio_inputs(cmd=cmd, strobe=0)
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = uio_inputs(cmd=cmd, strobe=1)
    # Wait for the done pulse; bounded so a hang fails loudly. DONE_DELAY is
    # 18 in v2 (sized for EMIT+hard-swish), hence the wider window than v1.
    seen_done = False
    for _ in range(28):
        await ClockCycles(dut.clk, 1)
        if done_flag(dut):
            seen_done = True
            break
    assert seen_done, f"no done pulse for cmd={cmd}"
    dut.uio_in.value = uio_inputs(cmd=cmd, strobe=0)
    await ClockCycles(dut.clk, 2)


async def read_acc(dut):
    """Stream the 32-bit accumulator out byte-by-byte and reassemble (signed)."""
    val = 0
    for i in range(4):
        dut.uio_in.value = uio_inputs(rd_sel=i)
        await ClockCycles(dut.clk, 1)
        val |= (int(dut.uo_out.value) & 0xFF) << (8 * i)
    if val >= 2**31:
        val -= 2**32
    return val


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
@cocotb.test()
async def test_reset_zero(dut):
    await reset(dut)
    assert await read_acc(dut) == 0
    assert ovf_flag(dut) == 0


@cocotb.test()
async def test_basic_mac(dut):
    await reset(dut)
    await op(dut, CMD_LOADW, 3)
    await op(dut, CMD_MAC, 5)
    assert await read_acc(dut) == 15
    await op(dut, CMD_MAC, 5)
    assert await read_acc(dut) == 30
    await op(dut, CMD_CLEAR)
    assert await read_acc(dut) == 0


@cocotb.test()
async def test_signed(dut):
    await reset(dut)
    await op(dut, CMD_LOADW, -4)
    await op(dut, CMD_MAC, 7)        # -28
    assert await read_acc(dut) == -28
    await op(dut, CMD_MAC, -3)       # -28 + 12 = -16
    assert await read_acc(dut) == -16
    # New weight is stationary across subsequent activations.
    await op(dut, CMD_LOADW, -128)
    await op(dut, CMD_MAC, -128)     # -16 + 16384 = 16368
    assert await read_acc(dut) == 16368


@cocotb.test()
async def test_clear_resets_ovf(dut):
    await reset(dut)
    await op(dut, CMD_LOADW, 100)
    await op(dut, CMD_MAC, 100)      # 10000, no overflow
    assert ovf_flag(dut) == 0
    await op(dut, CMD_CLEAR)
    assert await read_acc(dut) == 0
    assert ovf_flag(dut) == 0


@cocotb.test()
async def test_v2_opcodes_dont_touch_mac_state(dut):
    """LOAD_CFG/EMIT must leave MAC state untouched — and EMIT (101) must NOT
    alias into LOADW (01) inside the core (the cmd[2] gate). The NOP after
    EMIT re-arms the accumulator view on uo_out (EMIT switches it to the
    result register)."""
    await reset(dut)
    await op(dut, CMD_LOADW, 11)
    await op(dut, CMD_MAC, 10)               # acc = 110
    assert await read_acc(dut) == 110
    await op(dut, CMD_LOAD_CFG, 0x55)        # done pulses (asserted inside op)
    await op(dut, CMD_NOP)                   # restart cfg pointer after stray write
    await op(dut, CMD_EMIT, 77)              # would clobber weight if aliased
    await op(dut, CMD_NOP)                   # back to accumulator view
    assert await read_acc(dut) == 110        # acc untouched
    await op(dut, CMD_MAC, 10)               # weight must still be 11
    assert await read_acc(dut) == 220


@cocotb.test()
async def test_random_sequence(dut):
    await reset(dut)
    gold = GoldenPe()
    rng = random.Random(7)
    await op(dut, CMD_CLEAR)
    gold.clear()
    for _ in range(150):
        choice = rng.random()
        if choice < 0.25:
            w = rng.randint(-128, 127)
            await op(dut, CMD_LOADW, w)
            gold.load_w(w)
        elif choice < 0.95:
            a = rng.randint(-128, 127)
            await op(dut, CMD_MAC, a)
            gold.mac(a)
        else:
            await op(dut, CMD_CLEAR)
            gold.clear()
        assert await read_acc(dut) == gold.acc, f"acc mismatch, expected {gold.acc}"
        assert ovf_flag(dut) == gold.ovf


# ----------------------------------------------------------------------------
# P2: LOAD_CFG + EMIT (requantize + activation)
# ----------------------------------------------------------------------------
async def load_cfg(dut, gold):
    """Push the golden model's current config into the DUT byte-by-byte.
    A leading NOP resets the cfg write pointer (the pe_cfg contract)."""
    await op(dut, CMD_NOP)
    for b in gold.cfg_bytes():
        await op(dut, CMD_LOAD_CFG, b if b < 128 else b - 256)


async def read_result(dut):
    """uo_out shows the EMIT result until the next strobed command."""
    await ClockCycles(dut.clk, 1)
    v = int(dut.uo_out.value) & 0xFF
    return v - 256 if v >= 128 else v


@cocotb.test()
async def test_emit_basic_requant(dut):
    """Known config, small accumulation, clamp activation = passthrough."""
    await reset(dut)
    gold = GoldenPe()
    gold.m0, gold.n = quantize_multiplier_q15(0.05)
    gold.zp = -10
    await load_cfg(dut, gold)
    await op(dut, CMD_LOADW, 25)
    gold.load_w(25)
    for a in (40, 40, 33):
        await op(dut, CMD_MAC, a)
        gold.mac(a)
    await op(dut, CMD_EMIT)
    assert await read_result(dut) == gold.emit()   # 113*25*0.05 - 10 ~= 131 -> clamps
    # result view drops back to the accumulator on the next command...
    await op(dut, CMD_NOP)
    assert await read_acc(dut) == gold.acc
    # ...and EMIT is repeatable: acc was not consumed.
    await op(dut, CMD_EMIT)
    assert await read_result(dut) == gold.emit()


@cocotb.test()
async def test_emit_relu_relu6_bounds(dut):
    """ReLU/ReLU6 as clamp bounds, negative accumulations clamp to zp."""
    await reset(dut)
    gold = GoldenPe()
    gold.m0, gold.n = quantize_multiplier_q15(0.013)
    gold.zp = -10
    gold.qmin = gold.zp              # fused ReLU
    gold.qmax = 127
    await load_cfg(dut, gold)
    await op(dut, CMD_LOADW, -100)
    gold.load_w(-100)
    await op(dut, CMD_MAC, 100)      # acc = -10000 -> well below zero
    gold.mac(100)
    await op(dut, CMD_EMIT)
    assert await read_result(dut) == gold.emit() == gold.zp


@cocotb.test()
async def test_random_layers_vs_golden(dut):
    """Randomized end-to-end micro-layers: cfg + MACs + EMIT vs GoldenPe."""
    await reset(dut)
    rng = random.Random(11)
    for trial in range(25):
        gold = GoldenPe()
        gold.m0, gold.n = quantize_multiplier_q15(10 ** rng.uniform(-4, -1e-4))
        gold.zp = rng.randint(-128, 127)
        lo, hi = sorted((rng.randint(-128, 127), rng.randint(-128, 127)))
        gold.qmin, gold.qmax = lo, hi
        gold.act = ACT_CLAMP
        await load_cfg(dut, gold)
        await op(dut, CMD_CLEAR)
        gold.clear()
        w = rng.choice([rng.randint(-128, 127), -128, 127])
        await op(dut, CMD_LOADW, w)
        gold.load_w(w)
        for _ in range(rng.randint(1, 10)):
            a = rng.choice([rng.randint(-128, 127), -128, 127])
            await op(dut, CMD_MAC, a)
            gold.mac(a)
        await op(dut, CMD_EMIT)
        got = await read_result(dut)
        want = gold.emit()
        assert got == want, (
            f"trial {trial}: emit {got} != golden {want} "
            f"(acc={gold.acc} m0={gold.m0} n={gold.n} zp={gold.zp} "
            f"qmin={gold.qmin} qmax={gold.qmax})"
        )


@cocotb.test()
async def test_hardswish_sweep_vs_golden(dut):
    """Hard-swish across the full int8 input range at three scales.

    Identity requant trick: acc = 2*(q - zp) with m0 = 2^14, n = 0 makes the
    linear point land exactly on q (no rounding ambiguity), so the sweep
    isolates the activation datapath."""
    await reset(dut)
    for scale, zp in [(3 / 127, 0), (0.05, -20), (0.043, 5)]:
        gold = GoldenPe()
        gold.m0, gold.n = 1 << 14, 0
        gold.zp = zp
        gold.act = ACT_HSWISH
        for k, v in hswish_params(scale, zp).items():
            setattr(gold, k, v)
        await load_cfg(dut, gold)
        await op(dut, CMD_LOADW, 1)
        gold.load_w(1)
        for q in range(-128, 128, 3):       # step 3: 86 points per scale
            target = 2 * (q - zp)           # |target| <= 510
            await op(dut, CMD_CLEAR)
            gold.clear()
            rem = target                    # accumulate in int8-sized chunks
            while rem != 0:
                chunk = max(-128, min(127, rem))
                await op(dut, CMD_MAC, chunk)
                gold.mac(chunk)
                rem -= chunk
            await op(dut, CMD_EMIT)
            got = await read_result(dut)
            want = gold.emit()
            assert got == want, (
                f"hswish scale={scale} zp={zp} q={q}: {got} != {want}"
            )


@cocotb.test()
async def test_cfg_pointer_resets_on_other_commands(dut):
    """A partial cfg write followed by any other command restarts the pointer
    at byte 0 — a stale pointer would corrupt M0 and skew the EMIT result."""
    await reset(dut)
    gold = GoldenPe()
    gold.m0, gold.n = quantize_multiplier_q15(0.25)
    gold.zp = 7
    # Write garbage into the first three cfg slots, then abandon mid-block.
    await op(dut, CMD_NOP)
    for b in (0x12, 0x34, 0x1F):
        await op(dut, CMD_LOAD_CFG, b)
    # The full reload must land at byte 0 again (load_cfg NOPs first).
    await load_cfg(dut, gold)
    await op(dut, CMD_LOADW, 64)
    gold.load_w(64)
    await op(dut, CMD_MAC, 64)
    gold.mac(64)
    await op(dut, CMD_EMIT)
    assert await read_result(dut) == gold.emit()


@cocotb.test(skip=_SKIP_SLOW)
async def test_positive_saturation(dut):
    # weight=-128, data=-128 -> +16384 per MAC. 2^31 / 2^14 = 131072 MACs lands
    # exactly on +2^31, one past INT32_MAX, so the final add must clamp + flag.
    await reset(dut)
    gold = GoldenPe()
    await op(dut, CMD_CLEAR)
    gold.clear()
    await op(dut, CMD_LOADW, -128)
    gold.load_w(-128)

    base_mac = uio_inputs(cmd=CMD_MAC)
    dut.ui_in.value = s8_bits(-128)
    # 7-cycle period (2 high / 5 low) keeps ops spaced wider than the 4-deep
    # pipeline so each MAC fully commits before the next one is issued.
    N = 131072
    for _ in range(N):
        dut.uio_in.value = base_mac | (1 << 3)   # strobe high
        await ClockCycles(dut.clk, 2)
        dut.uio_in.value = base_mac              # strobe low (re-arm)
        await ClockCycles(dut.clk, 5)
        gold.mac(-128)

    assert gold.acc == INT32_MAX               # golden sanity: it saturated
    assert await read_acc(dut) == INT32_MAX
    assert ovf_flag(dut) == 1


@cocotb.test(skip=_SKIP_SLOW)
async def test_negative_saturation(dut):
    # weight=127, data=-128 -> -16256 per MAC. ~132137 MACs cross -2^31.
    await reset(dut)
    gold = GoldenPe()
    await op(dut, CMD_CLEAR)
    gold.clear()
    await op(dut, CMD_LOADW, 127)
    gold.load_w(127)

    base_mac = uio_inputs(cmd=CMD_MAC)
    dut.ui_in.value = s8_bits(-128)
    # Enough iterations to cross -2^31: ceil(2^31 / 16256) = 132137.
    N = 132137
    for _ in range(N):
        dut.uio_in.value = base_mac | (1 << 3)
        await ClockCycles(dut.clk, 2)
        dut.uio_in.value = base_mac
        await ClockCycles(dut.clk, 5)
        gold.mac(-128)

    assert gold.acc == INT32_MIN
    assert await read_acc(dut) == INT32_MIN
    assert ovf_flag(dut) == 1

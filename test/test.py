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
from pe_golden import INT32_MAX, INT32_MIN, GoldenPe  # noqa: E402

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
    # Wait for the done pulse; bounded so a hang fails loudly.
    seen_done = False
    for _ in range(16):
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
async def test_v2_opcodes_inert_in_p1(dut):
    """LOAD_CFG/EMIT must pulse done but leave MAC state untouched — and EMIT
    (101) must NOT alias into LOADW (01) inside the core (the cmd[2] gate)."""
    await reset(dut)
    await op(dut, CMD_LOADW, 11)
    await op(dut, CMD_MAC, 10)               # acc = 110
    assert await read_acc(dut) == 110
    await op(dut, CMD_LOAD_CFG, 0x55)        # done pulses (asserted inside op)
    await op(dut, CMD_EMIT, 77)              # would clobber weight if aliased
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

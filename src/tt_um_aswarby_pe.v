/*
 * tt_um_aswarby_pe - Tiny Tapeout top wrapper (v2: INT8 PE)
 *
 * Byte-serial, weight-stationary INT8 processing element. P1 scope: the v1
 * MAC engine (mac_core/mac_fsm, ported verbatim from tt-aswarby-mac) behind
 * the widened v2 command interface. The two v2 opcodes (LOAD_CFG, EMIT) are
 * decoded but inert until P2 lands the requantize datapath; they complete
 * with `done` like every strobed command so the host contract never changes.
 *
 * The arithmetic contract is frozen in model/pe_golden.py — RTL follows it,
 * never the other way round.
 *
 * Pin map (v2 — see docs/PLAN.md section 1.4)
 * -------------------------------------------
 *   ui_in[7:0]   data byte in (signed INT8: weight, activation, or cfg byte)
 *   uio_in[2:0]  command  : 000 NOP / 001 load weight / 010 MAC / 011 clear
 *                           100 LOAD_CFG (P2) / 101 EMIT (P2) / 11x reserved
 *   uio_in[3]    strobe   : rising edge executes one command
 *   uio_in[5:4]  rd_sel   : selects which accumulator byte appears on uo_out
 *   uo_out[7:0]  selected accumulator byte (read 4 bytes LSB-first for INT32)
 *   uio_out[6]   done     : one-cycle completion pulse
 *   uio_out[7]   ovf      : sticky saturation flag (cleared by clear/reset)
 *
 * Copyright (c) 2026 Mark Shilton
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_aswarby_pe (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (1=output, 0=input)
    input  wire       ena,      // high while the design is selected
    input  wire       clk,      // clock
    input  wire       rst_n     // active-low reset
);

  wire [2:0] cmd    = uio_in[2:0];
  wire       strobe = uio_in[3];
  wire [1:0] rd_sel = uio_in[5:4];

  wire       do_op;
  wire       done;
  wire       ovf;
  wire [7:0] rd_byte;

  mac_fsm u_fsm (
      .clk   (clk),
      .rst_n (rst_n),
      .strobe(strobe),
      .do_op (do_op),
      .done  (done)
  );

  // cmd[1:0] of the v2 opcodes 000..011 match the v1 encoding exactly, so the
  // core is reused untouched. Gating do_op on !cmd[2] keeps the v2 opcodes
  // (LOAD_CFG=100, EMIT=101) from aliasing into LOADW/NOP inside the core.
  mac_core u_core (
      .clk    (clk),
      .rst_n  (rst_n),
      .do_op  (do_op && !cmd[2]),
      .cmd    (cmd[1:0]),
      .data   (ui_in),
      .rd_sel (rd_sel),
      .rd_byte(rd_byte),
      .ovf    (ovf)
  );

  assign uo_out = rd_byte;

  // Bidirectional bus: bits 7,6 are outputs; bits 5..0 are inputs.
  assign uio_out = {ovf, done, 6'b00_0000};
  assign uio_oe  = 8'b1100_0000;

  // Silence unused-signal warnings (ena and the spare uio inputs).
  wire _unused = &{ena, uio_in[7:6], 1'b0};

endmodule

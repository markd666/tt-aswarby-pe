/*
 * tt_um_aswarby_pe - Tiny Tapeout top wrapper (v2: INT8 PE)
 *
 * Byte-serial, weight-stationary INT8 processing element: the v1 MAC engine
 * (mac_core/mac_fsm, ported from tt-aswarby-mac) plus the v2 post-accumulation
 * stage — Q15 requantize and fused activations (pe_cfg + pe_requant).
 *
 * The arithmetic contract is frozen in model/pe_golden.py — RTL follows it,
 * never the other way round.
 *
 * Pin map (v2 — see docs/PLAN.md section 1.4)
 * -------------------------------------------
 *   ui_in[7:0]   data byte in (signed INT8: weight, activation, or cfg byte)
 *   uio_in[2:0]  command  : 000 NOP / 001 load weight / 010 MAC / 011 clear
 *                           100 LOAD_CFG / 101 EMIT / 11x reserved
 *   uio_in[3]    strobe   : rising edge executes one command
 *   uio_in[5:4]  rd_sel   : selects which accumulator byte appears on uo_out
 *   uo_out[7:0]  accumulator byte view (rd_sel, LSB-first) — switches to the
 *                EMIT result after an EMIT completes, and back to the
 *                accumulator view on the next strobed command of any kind
 *   uio_out[6]   done     : one-cycle completion pulse (fixed latency, all ops)
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

  wire is_cfg  = (cmd == 3'b100);
  wire is_emit = (cmd == 3'b101);

  wire        do_op;
  wire        done;
  wire        ovf;
  wire [7:0]  rd_byte;
  wire [31:0] acc;

  // DONE_DELAY covers the slowest operation: EMIT with hard-swish writes its
  // result 16 cycles after do_op (pe_requant header); 18 adds safe margin.
  mac_fsm #(
      .DONE_DELAY(18)
  ) u_fsm (
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
      .acc_out(acc),
      .ovf    (ovf)
  );

  wire [15:0] cfg_m0, cfg_mhs;
  wire [4:0]  cfg_n, cfg_nhs;
  wire        cfg_act;
  wire [7:0]  cfg_zp, cfg_qmin, cfg_qmax, cfg_q3, cfg_q6;

  pe_cfg u_cfg (
      .clk   (clk),
      .rst_n (rst_n),
      .do_op (do_op),
      .is_cfg(is_cfg),
      .data  (ui_in),
      .m0    (cfg_m0),
      .n     (cfg_n),
      .act   (cfg_act),
      .zp    (cfg_zp),
      .qmin  (cfg_qmin),
      .qmax  (cfg_qmax),
      .q3    (cfg_q3),
      .q6    (cfg_q6),
      .mhs   (cfg_mhs),
      .nhs   (cfg_nhs)
  );

  wire [7:0] result;
  wire       result_we;

  pe_requant u_requant (
      .clk      (clk),
      .rst_n    (rst_n),
      .start    (do_op && is_emit),
      .acc      (acc),
      .m0       (cfg_m0),
      .n        (cfg_n),
      .act      (cfg_act),
      .zp       (cfg_zp),
      .qmin     (cfg_qmin),
      .qmax     (cfg_qmax),
      .q3       (cfg_q3),
      .q6       (cfg_q6),
      .mhs      (cfg_mhs),
      .nhs      (cfg_nhs),
      .result   (result),
      .result_we(result_we)
  );

  // Output view: accumulator bytes by default; the EMIT result from the cycle
  // it is written until the next strobed command (of any kind) re-arms the
  // accumulator view. result_we and do_op can never coincide (EMIT result
  // lands mid-flight, do_op only fires from an idle, re-armed handshake).
  reg show_result;
  always @(posedge clk) begin
    if (!rst_n)         show_result <= 1'b0;
    else if (do_op)     show_result <= 1'b0;
    else if (result_we) show_result <= 1'b1;
  end

  assign uo_out = show_result ? result : rd_byte;

  // Bidirectional bus: bits 7,6 are outputs; bits 5..0 are inputs.
  assign uio_out = {ovf, done, 6'b00_0000};
  assign uio_oe  = 8'b1100_0000;

  // Silence unused-signal warnings (ena and the spare uio inputs).
  wire _unused = &{ena, uio_in[7:6], 1'b0};

endmodule

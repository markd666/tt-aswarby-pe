/*
 * pe_cfg - configuration register file with auto-incrementing write pointer
 *
 * Eleven byte-wide registers loaded via the LOAD_CFG command. The byte layout
 * is canonical in model/pe_golden.py (GoldenPe.cfg_bytes) — keep them in sync:
 *
 *   0: M0[7:0]    1: M0[15:8]    2: {act, n[4:0]} (act = bit 5)
 *   3: ZP         4: QMIN        5: QMAX
 *   6: Q3         7: Q6          8: MHS[7:0]
 *   9: MHS[15:8] 10: NHS[4:0]
 *
 * Pointer contract: resets to 0 on reset and on any executed command that is
 * NOT LOAD_CFG, so the host always writes a config block from the top —
 * e.g. NOP, then eleven LOAD_CFG strobes. After byte 10 the pointer wraps
 * to 0 (a 12th consecutive write overwrites M0[7:0]).
 *
 * Copyright (c) 2026 Mark Shilton
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module pe_cfg (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        do_op,     // one-cycle execute pulse (any command)
    input  wire        is_cfg,    // command == LOAD_CFG
    input  wire [7:0]  data,      // config byte from ui_in
    output wire [15:0] m0,        // Q15 requantize mantissa
    output wire [4:0]  n,         // extra right shift (total = 15 + n)
    output wire        act,       // 0 = clamp activation, 1 = hard-swish
    output wire [7:0]  zp,        // output zero-point (signed)
    output wire [7:0]  qmin,      // clamp lower bound (signed)
    output wire [7:0]  qmax,      // clamp upper bound (signed)
    output wire [7:0]  q3,        // hard-swish knee: quantize(3.0) (unsigned)
    output wire [7:0]  q6,        // hard-swish knee: quantize(6.0) (unsigned)
    output wire [15:0] mhs,       // hard-swish Q15 mantissa
    output wire [4:0]  nhs        // hard-swish extra right shift
);

  reg [3:0] ptr;
  reg [7:0] cfg [0:10];
  integer i;

  always @(posedge clk) begin
    if (!rst_n) begin
      ptr <= 4'd0;
      // Reset values mirror GoldenPe.__init__: m0 = 0.5 in Q15, full-range
      // clamp, everything else zero.
      cfg[0]  <= 8'h00;  // M0 lo
      cfg[1]  <= 8'h40;  // M0 hi (0x4000 = 2^14)
      cfg[2]  <= 8'h00;  // act=0, n=0
      cfg[3]  <= 8'h00;  // ZP
      cfg[4]  <= 8'h80;  // QMIN = -128
      cfg[5]  <= 8'h7F;  // QMAX = +127
      cfg[6]  <= 8'h00;  // Q3
      cfg[7]  <= 8'h00;  // Q6
      cfg[8]  <= 8'h00;  // MHS lo
      cfg[9]  <= 8'h40;  // MHS hi (0x4000 = 2^14)
      cfg[10] <= 8'h00;  // NHS
    end else if (do_op) begin
      if (is_cfg) begin
        cfg[ptr] <= data;
        ptr      <= (ptr == 4'd10) ? 4'd0 : ptr + 4'd1;
      end else begin
        ptr <= 4'd0;
      end
    end
  end

  assign m0   = {cfg[1], cfg[0]};
  assign n    = cfg[2][4:0];
  assign act  = cfg[2][5];
  assign zp   = cfg[3];
  assign qmin = cfg[4];
  assign qmax = cfg[5];
  assign q3   = cfg[6];
  assign q6   = cfg[7];
  assign mhs  = {cfg[9], cfg[8]};
  assign nhs  = cfg[10][4:0];

endmodule

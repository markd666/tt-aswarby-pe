/*
 * mac_core - signed INT8 x INT8 -> INT32 weight-stationary multiply-accumulate
 *
 * Pipelined datapath (4 stages) so each stage's logic stays short enough to
 * close timing at 50 MHz on the slow 180 nm GF180 node, where even a single
 * 8x8 multiply is too long a combinational path for 20 ns:
 *
 *   stage 1  split-multiply : pp_hi <= a_hi * weight ; pp_lo <= a_lo * weight
 *   stage 2  reconstruct    : prod_q <= (pp_hi << 4) + pp_lo          (full product)
 *   stage 3  accumulate     : sum_q  <= acc + prod_q                  (33-bit add)
 *   stage 4  saturate       : acc    <= fits ? sum_q : clamp          (sign-bit test)
 *
 * The activation byte a is split as a = a_hi*16 + a_lo with a_hi a signed
 * nibble (-8..7) and a_lo an unsigned nibble (0..15), so a*weight =
 * (a_hi*weight)<<4 + a_lo*weight. Each half is a 4x8 multiply (about half the
 * depth of an 8x8), parallel in stage 1 and summed in stage 2.
 *
 * A MAC commits to `acc` four cycles after its `do_op`. Operations are issued
 * one at a time via the strobe handshake (mac_fsm), spaced wider than the
 * pipeline depth, so no forwarding/hazard logic is needed.
 *
 * Copyright (c) 2026 Mark Shilton
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module mac_core (
    input  wire        clk,
    input  wire        rst_n,    // active-low reset
    input  wire        do_op,    // one-cycle strobe: accept `cmd` this cycle
    input  wire [1:0]  cmd,      // 00 NOP, 01 load weight, 10 MAC, 11 clear
    input  wire [7:0]  data,     // signed INT8 operand (weight or activation)
    input  wire [1:0]  rd_sel,   // which accumulator byte appears on rd_byte
    output wire [7:0]  rd_byte,  // selected accumulator byte (combinational)
    output wire [31:0] acc_out,  // full accumulator (v2: feeds pe_requant EMIT)
    output wire        ovf       // sticky: accumulator has saturated since clear
);

  localparam [1:0] CMD_NOP   = 2'b00,
                   CMD_LOADW = 2'b01,
                   CMD_MAC   = 2'b10,
                   CMD_CLEAR = 2'b11;

  reg signed [7:0]  weight_q;
  reg signed [31:0] acc;
  reg               ovf_sticky;

  // Pipeline valids and registers.
  reg               mac_v1, mac_v2, mac_v3;
  reg signed [12:0] pp_hi, pp_lo;   // stage 1: half products
  reg signed [16:0] prod_q;         // stage 2: full product (fits in 16b; 17b spare)
  reg signed [32:0] sum_q;          // stage 3: pre-saturate sum (33-bit headroom)

  // Activation nibble split (combinational off the input byte).
  wire signed [3:0] a_hi = data[7:4];                 // signed nibble (-8..7)
  wire signed [5:0] a_lo = $signed({2'b00, data[3:0]}); // unsigned nibble as +ve signed

  always @(posedge clk) begin
    if (!rst_n) begin
      weight_q   <= 8'sd0;
      acc        <= 32'sd0;
      ovf_sticky <= 1'b0;
      mac_v1     <= 1'b0;
      mac_v2     <= 1'b0;
      mac_v3     <= 1'b0;
      pp_hi      <= 13'sd0;
      pp_lo      <= 13'sd0;
      prod_q     <= 17'sd0;
      sum_q      <= 33'sd0;
    end else begin
      // Valids advance every cycle; default no new MAC.
      mac_v1 <= 1'b0;
      mac_v2 <= mac_v1;
      mac_v3 <= mac_v2;

      // ---- stage 0/1: accept op; split-multiply -----------------------------
      if (do_op) begin
        case (cmd)
          CMD_LOADW: weight_q <= data;          // reinterpret byte as INT8
          CMD_MAC: begin
            pp_hi  <= a_hi * weight_q;           // signed 4x8
            pp_lo  <= a_lo * weight_q;           // signed 6x8 (a_lo >= 0)
            mac_v1 <= 1'b1;
          end
          CMD_CLEAR: begin
            acc        <= 32'sd0;
            ovf_sticky <= 1'b0;
          end
          default: ; // NOP
        endcase
      end

      // ---- stage 2: reconstruct full product --------------------------------
      if (mac_v1)
        prod_q <= (pp_hi <<< 4) + pp_lo;

      // ---- stage 3: 33-bit accumulate ---------------------------------------
      if (mac_v2)
        sum_q <= {acc[31], acc} + {{16{prod_q[16]}}, prod_q};

      // ---- stage 4: saturate (overflow iff top two bits differ) -------------
      if (mac_v3) begin
        if (sum_q[32] != sum_q[31]) begin
          acc        <= sum_q[32] ? 32'sh8000_0000 : 32'sh7FFF_FFFF;
          ovf_sticky <= 1'b1;
        end else begin
          acc <= sum_q[31:0];
        end
      end
    end
  end

  assign rd_byte = (rd_sel == 2'd0) ? acc[7:0]   :
                   (rd_sel == 2'd1) ? acc[15:8]  :
                   (rd_sel == 2'd2) ? acc[23:16] :
                                      acc[31:24];

  assign acc_out = acc;
  assign ovf = ovf_sticky;

endmodule

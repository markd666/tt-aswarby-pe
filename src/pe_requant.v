/*
 * pe_requant - EMIT engine: Q15 requantize + fused activation (sequential)
 *
 * Implements GoldenPe.emit() from model/pe_golden.py bit-for-bit:
 *
 *   r    = round_half_away( acc * M0  >>  (15 + n) )
 *   q    = clamp( r + ZP, -128, 127 )
 *   out  = clamp-first(q, QMIN, QMAX)                      (ACT_CLAMP)
 *        | clamp( rha(u*v*MHS >> 15+NHS) + ZP, -128, 127 ) (ACT_HSWISH)
 *          where u = q - ZP,  v = clamp(u + Q3, 0, Q6)
 *
 * Micro-architecture: fully SEQUENTIAL. The first version of this module
 * instantiated two parallel 17x17 multipliers and a 48-bit barrel shifter;
 * yosys maps a single 17x17 to ~2,000 cells, and the module hardened at
 * 241% utilization of the 1x2 tile. Operations are strobe-paced by the
 * host, so multiplier latency is worthless here — everything now runs
 * through ONE 48-bit adder:
 *
 *   - multiply  : 16 shift-add iterations (M examined LSB-first, A doubling)
 *   - rounding  : (|t| + 2^(s-1)) >> s computed as |t| >> (s-1), +1, >> 1
 *                 (exact identity, no bias decoder / barrel shifter needed)
 *   - negate    : through the same adder (0 - x)
 *
 * The engine runs once for the requantize (acc x M0) and, for hard-swish,
 * twice more (u x v, then p x MHS). Worst-case EMIT latency is ~180 cycles
 * (~3.6 us at 50 MHz) — irrelevant against an RP2040 host strobing commands.
 * `done` for EMIT is completion-based (top level), not a fixed delay.
 * `acc` and the overflow flag are NOT modified — EMIT is repeatable.
 *
 * Copyright (c) 2026 Mark Shilton
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module pe_requant (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,     // one-cycle pulse: begin EMIT of `acc`
    input  wire [31:0] acc,       // accumulator (signed)
    input  wire [15:0] m0,
    input  wire [4:0]  n,
    input  wire        act,       // 0 clamp / 1 hard-swish
    input  wire [7:0]  zp,
    input  wire [7:0]  qmin,
    input  wire [7:0]  qmax,
    input  wire [7:0]  q3,
    input  wire [7:0]  q6,
    input  wire [15:0] mhs,
    input  wire [4:0]  nhs,
    output reg  [7:0]  result,    // int8 output (signed), valid at result_we
    output reg         result_we  // one-cycle pulse when `result` is written
);

  // ---- sequencer ------------------------------------------------------------
  localparam [3:0] S_IDLE   = 4'd0,
                   S_MUL    = 4'd1,   // 16 shift-add iterations: t += m[0]?a:0
                   S_ABS    = 4'd2,   // t < 0 ? -t : t (sign remembered)
                   S_SHR    = 4'd3,   // (s-1) arithmetic right shifts of |t|
                   S_INC    = 4'd4,   // +1 (the rounding-half bit)
                   S_SHR1   = 4'd5,   // final >> 1
                   S_SIGN   = 4'd6,   // restore sign -> r
                   S_Q      = 4'd7,   // q = clamp(r+zp); clamp-act finishes here
                   S_UV     = 4'd8,   // hard-swish: u, v
                   S_LOADP  = 4'd9,   // start u*v multiply (pass: P)
                   S_LOADHS = 4'd10,  // start p*MHS multiply (pass: HS)
                   S_OUT    = 4'd11;  // result = clamp(r2 + zp)

  localparam [1:0] PASS_REQ = 2'd0,   // acc * M0   >> 15+n
                   PASS_P   = 2'd1,   // u * v      (no shift, raw product)
                   PASS_HS  = 2'd2;   // p * MHS    >> 15+nhs

  reg [3:0] state;
  reg [1:0] pass;

  // ---- engine registers -------------------------------------------------------
  reg signed [47:0] t_q;     // accumulator / working register
  reg signed [47:0] a_sh;    // multiplicand, doubling each iteration
  reg        [15:0] m_sh;    // multiplier bits, LSB-first
  reg        [3:0]  mcnt;    // multiply iteration counter (16 per pass)
  reg        [5:0]  scnt;    // shift counter (s-1 right shifts)
  reg               tneg;    // sign of t, applied after rounding

  // ---- activation registers -----------------------------------------------------
  reg signed [7:0]  q_lin;   // clamp(r + zp, -128, 127)
  reg signed [9:0]  u_q;     // q - zp  in [-255, 255]
  reg        [7:0]  v_q;     // clamp(u + q3, 0, q6)
  reg signed [17:0] p_q;     // u * v   in [-65025, 65025]

  // ---- shared combinational helpers ----------------------------------------------
  wire signed [47:0] t_add = t_q + a_sh;          // the one real adder
  wire signed [47:0] t_neg = -t_q;

  // r = +/- t after rounding; |r| < 2^33 by construction (s >= 15)
  wire signed [34:0] r_w  = tneg ? -$signed({1'b0, t_q[33:0]})
                                 :  $signed({1'b0, t_q[33:0]});

  // q = clamp(r + zp, -128, 127). $signed() on the concat is load-bearing: a
  // bare concat is unsigned and would force the whole addition unsigned,
  // zero-extending a negative r into a huge positive value.
  wire signed [35:0] r_zp = r_w + $signed({{28{zp[7]}}, zp});
  wire signed [7:0]  q_w  = (r_zp < -36'sd128) ? -8'sd128 :
                            (r_zp >  36'sd127) ?  8'sd127 : r_zp[7:0];

  // clamp activation: lower bound checked first (matches GoldenPe.clamp)
  wire signed [7:0] qmin_s = qmin;
  wire signed [7:0] qmax_s = qmax;
  wire signed [7:0] out_clamp = (q_w < qmin_s) ? qmin_s :
                                (q_w > qmax_s) ? qmax_s : q_w;

  // hard-swish helpers (width-exact: bit patterns are signedness-safe)
  wire signed [9:0]  u_w = {{2{q_lin[7]}}, q_lin} - {{2{zp[7]}}, zp};
  wire signed [10:0] uq3 = {u_w[9], u_w} + {3'b000, q3};
  wire        [7:0]  v_w = uq3[10]              ? 8'd0 :
                           (uq3 > {3'b000, q6}) ? q6   : uq3[7:0];

  // shift count for the current scaling pass (s = 15 + n; we shift s-1, +1, >>1)
  wire [5:0] s_total = 6'd15 + {1'b0, (pass == PASS_REQ) ? n : nhs};

  always @(posedge clk) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      pass      <= PASS_REQ;
      result    <= 8'd0;
      result_we <= 1'b0;
      t_q       <= 48'sd0;
      a_sh      <= 48'sd0;
      m_sh      <= 16'd0;
      mcnt      <= 4'd0;
      scnt      <= 6'd0;
      tneg      <= 1'b0;
      q_lin     <= 8'sd0;
      u_q       <= 10'sd0;
      v_q       <= 8'd0;
      p_q       <= 18'sd0;
    end else begin
      result_we <= 1'b0;   // default one-shot

      case (state)
        S_IDLE: if (start) begin
          t_q   <= 48'sd0;
          a_sh  <= {{16{acc[31]}}, acc};
          m_sh  <= m0;
          mcnt  <= 4'd0;
          pass  <= PASS_REQ;
          state <= S_MUL;
        end

        S_MUL: begin
          if (m_sh[0])
            t_q <= t_add;
          a_sh <= a_sh <<< 1;
          m_sh <= m_sh >> 1;
          mcnt <= mcnt + 4'd1;
          if (mcnt == 4'd15)
            state <= (pass == PASS_P) ? S_LOADHS : S_ABS;
        end

        S_ABS: begin
          tneg  <= t_q[47];
          if (t_q[47])
            t_q <= t_neg;
          scnt  <= s_total - 6'd1;
          state <= S_SHR;
        end

        S_SHR: begin
          t_q  <= t_q >>> 1;     // |t| is non-negative; >>> keeps widths tidy
          scnt <= scnt - 6'd1;
          if (scnt == 6'd1)
            state <= S_INC;
        end

        S_INC: begin
          t_q   <= t_q + 48'sd1;
          state <= S_SHR1;
        end

        S_SHR1: begin
          t_q   <= t_q >>> 1;
          state <= S_SIGN;
        end

        S_SIGN: state <= S_Q;    // r_w mux applies the sign combinationally

        S_Q: begin
          if (pass == PASS_HS) begin
            // hard-swish pass complete: r is the scaled u*v*MHS
            result    <= q_w;          // clamp(r2 + zp, -128, 127)
            result_we <= 1'b1;
            state     <= S_IDLE;
          end else if (!act) begin
            result    <= out_clamp;    // ReLU / ReLU6 / passthrough bounds
            result_we <= 1'b1;
            state     <= S_IDLE;
          end else begin
            q_lin <= q_w;
            state <= S_UV;
          end
        end

        S_UV: begin
          u_q   <= u_w;
          v_q   <= v_w;
          state <= S_LOADP;
        end

        S_LOADP: begin
          // multiply u by v through the engine: A = u, M = v (zero-extended)
          t_q   <= 48'sd0;
          a_sh  <= {{38{u_q[9]}}, u_q};
          m_sh  <= {8'd0, v_q};
          mcnt  <= 4'd0;
          pass  <= PASS_P;
          state <= S_MUL;
        end

        S_LOADHS: begin
          // p = u*v just finished in t_q (raw, unshifted); scale it by MHS
          p_q   <= t_q[17:0];
          t_q   <= 48'sd0;
          a_sh  <= {{30{t_q[17]}}, t_q[17:0]};
          m_sh  <= mhs;
          mcnt  <= 4'd0;
          pass  <= PASS_HS;
          state <= S_MUL;
        end

        S_OUT: state <= S_IDLE;  // unreachable; retained for completeness

        default: state <= S_IDLE;
      endcase
    end
  end

  // p_q is kept only for waveform debugging; silence the unused warning.
  wire _unused = &{p_q, 1'b0};

endmodule

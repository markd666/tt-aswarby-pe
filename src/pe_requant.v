/*
 * pe_requant - EMIT engine: Q15 requantize + fused activation
 *
 * Implements GoldenPe.emit() from model/pe_golden.py bit-for-bit:
 *
 *   r    = round_half_away( acc * M0  >>  (15 + n) )
 *   q    = clamp( r + ZP, -128, 127 )
 *   out  = clamp-first(q, QMIN, QMAX)                      (ACT_CLAMP)
 *        | clamp( rha(u*v*MHS >> 15+NHS) + ZP, -128, 127 ) (ACT_HSWISH)
 *          where u = q - ZP,  v = clamp(u + Q3, 0, Q6)
 *
 * Micro-architecture: ONE shared 48-bit "scale unit" — split 16x16 multiplies
 * (one cycle), 48-bit combine (one cycle), negate+round-bias add (one cycle),
 * variable barrel shift (one cycle), sign restore (one cycle) — sequenced
 * twice for hard-swish: pass 1 scales acc by M0, pass 2 scales p = u*v
 * (sign-extended to 32 bits) by MHS through the exact same hardware. Each
 * stage is kept to roughly one adder/mux depth: the v1 lesson (pipeline the
 * multiply from day one) applied to the bigger 32x16.
 *
 * Latency from `start`: 7 cycles (clamp activation), 14 cycles (hard-swish).
 * The top-level DONE_DELAY (16) covers both; ops are strobe-paced so
 * throughput is irrelevant. `acc` and the overflow flag are NOT modified —
 * EMIT is repeatable with different configs against the same accumulation.
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
    // configuration (pe_cfg outputs, stable while an op is in flight)
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

  // ---- sequencer states ----------------------------------------------------
  localparam [3:0] S_IDLE  = 4'd0,
                   S_MUL   = 4'd1,   // scale-unit stage 1: half products
                   S_COMB  = 4'd2,   // stage 2: 48-bit combine
                   S_BIAS  = 4'd3,   // stage 3: |t| + rounding bias
                   S_SHIFT = 4'd4,   // stage 4: barrel shift
                   S_SIGN  = 4'd5,   // stage 5: sign restore -> r
                   S_Q     = 4'd6,   // q = clamp(r + zp, -128, 127); clamp-act finishes
                   S_UV    = 4'd7,   // hard-swish: u, v
                   S_P     = 4'd8,   // hard-swish: p = u*v; reload scale unit
                   S_OUT   = 4'd9;   // hard-swish: result = clamp(r2 + zp)

  reg [3:0] state;
  reg       pass2;     // 0 = scaling acc by M0, 1 = scaling p by MHS

  // ---- scale-unit operand registers -----------------------------------------
  reg signed [31:0] a_q;     // value being scaled
  reg        [15:0] m_q;     // Q15 mantissa
  reg        [5:0]  s_q;     // total right shift (15 + n), 15..46

  // ---- scale-unit pipeline registers ----------------------------------------
  reg signed [32:0] ph_q;    // A[31:16] * M   (17s x 17s)
  reg signed [32:0] pl_q;    // A[15:0]  * M   (zero-extended x 17s)
  reg signed [47:0] t_q;     // full product A*M  (|t| < 2^47)
  reg               tneg_q;  // sign of t, carried past the abs
  reg        [47:0] tb_q;    // |t| + (1 << (s-1))
  reg        [47:0] sh_q;    // tb >> s
  reg signed [33:0] r_q;     // signed rounded result (|r| < 2^33)

  // ---- activation registers --------------------------------------------------
  reg signed [7:0]  q_lin;   // clamp(r + zp, -128, 127)
  reg signed [9:0]  u_q;     // q - zp  in [-255, 255]
  reg        [7:0]  v_q;     // clamp(u + q3, 0, q6)
  reg signed [17:0] p_q;     // u * v   in [-65025, 65025]

  // ---- combinational helpers --------------------------------------------------
  wire signed [16:0] a_hi  = {a_q[31], a_q[31:16]};      // sign-extended high half
  wire signed [16:0] a_lo  = {1'b0, a_q[15:0]};          // unsigned low half
  wire signed [16:0] m_s   = {1'b0, m_q};                // mantissa, always positive

  wire signed [47:0] t_w   = (ph_q <<< 16) + pl_q;

  wire        [47:0] t_abs = t_q[47] ? (~t_q + 48'd1) : t_q;
  wire        [47:0] bias  = 48'd1 << (s_q - 6'd1);      // s >= 15, so s-1 >= 14

  wire signed [33:0] r_pos = $signed({1'b0, sh_q[32:0]});
  wire signed [33:0] r_w   = tneg_q ? -r_pos : r_pos;

  // q = clamp(r + zp, -128, 127). The $signed() on the concat is load-bearing:
  // a bare concatenation is unsigned and would force the whole addition
  // unsigned, zero-extending a negative r_q into a huge positive value.
  wire signed [34:0] r_zp  = r_q + $signed({{27{zp[7]}}, zp});
  wire signed [7:0]  q_w   = (r_zp < -35'sd128) ? -8'sd128 :
                             (r_zp >  35'sd127) ?  8'sd127 : r_zp[7:0];

  // clamp activation: lower bound checked first (matches GoldenPe.clamp)
  wire signed [7:0] qmin_s = qmin;
  wire signed [7:0] qmax_s = qmax;
  wire signed [7:0] out_clamp = (q_w < qmin_s) ? qmin_s :
                                (q_w > qmax_s) ? qmax_s : q_w;

  // hard-swish helpers
  wire signed [9:0]  u_w   = {{2{q_lin[7]}}, q_lin} - {{2{zp[7]}}, zp};
  wire signed [10:0] uq3   = {u_w[9], u_w} + {3'b000, q3};
  wire        [7:0]  v_w   = uq3[10]                       ? 8'd0 :
                             (uq3 > {3'b000, q6})          ? q6   : uq3[7:0];

  always @(posedge clk) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      pass2     <= 1'b0;
      result    <= 8'd0;
      result_we <= 1'b0;
      a_q       <= 32'sd0;
      m_q       <= 16'd0;
      s_q       <= 6'd0;
      ph_q      <= 33'sd0;
      pl_q      <= 33'sd0;
      t_q       <= 48'sd0;
      tneg_q    <= 1'b0;
      tb_q      <= 48'd0;
      sh_q      <= 48'd0;
      r_q       <= 34'sd0;
      q_lin     <= 8'sd0;
      u_q       <= 10'sd0;
      v_q       <= 8'd0;
      p_q       <= 18'sd0;
    end else begin
      result_we <= 1'b0;   // default one-shot

      case (state)
        S_IDLE: if (start) begin
          a_q   <= acc;
          m_q   <= m0;
          s_q   <= 6'd15 + {1'b0, n};
          pass2 <= 1'b0;
          state <= S_MUL;
        end

        S_MUL: begin
          ph_q  <= a_hi * m_s;
          pl_q  <= a_lo * m_s;
          state <= S_COMB;
        end

        S_COMB: begin
          t_q   <= t_w;
          state <= S_BIAS;
        end

        S_BIAS: begin
          tneg_q <= t_q[47];
          tb_q   <= t_abs + bias;
          state  <= S_SHIFT;
        end

        S_SHIFT: begin
          sh_q  <= tb_q >> s_q;
          state <= S_SIGN;
        end

        S_SIGN: begin
          r_q   <= r_w;
          state <= S_Q;
        end

        S_Q: begin
          if (pass2) begin
            // hard-swish pass 2 complete: r is the scaled u*v*MHS
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
          state <= S_P;
        end

        S_P: begin
          p_q   <= u_q * $signed({1'b0, v_q});
          state <= S_OUT;
        end

        S_OUT: begin
          // reload the scale unit for pass 2: p sign-extended to 32 bits
          a_q   <= {{14{p_q[17]}}, p_q};
          m_q   <= mhs;
          s_q   <= 6'd15 + {1'b0, nhs};
          pass2 <= 1'b1;
          state <= S_MUL;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

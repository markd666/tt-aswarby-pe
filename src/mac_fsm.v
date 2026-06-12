/*
 * mac_fsm - strobe handshake + completion timing for mac_core
 *
 * Turns each rising edge of `strobe` into a single-cycle `do_op` pulse (so one
 * strobe = exactly one operation, no repeat while held high) and raises `done`
 * once the pipelined result has committed to the accumulator.
 *
 * `done` is `do_op` delayed by DONE_DELAY cycles. The mac_core MAC pipeline is
 * 4 deep, so DONE_DELAY is sized to fire `done` the cycle after the accumulator
 * settles; the host waits for `done` before reading or issuing the next op,
 * which also guarantees ops are spaced wider than the pipeline depth.
 *
 * Note: `strobe` is assumed roughly synchronous to clk (driven by the Tiny
 * Tapeout Commander). A genuinely asynchronous source would get a two-flop
 * synchroniser ahead of this FSM.
 *
 * Copyright (c) 2026 Mark Shilton
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module mac_fsm #(
    parameter DONE_DELAY = 5
) (
    input  wire clk,
    input  wire rst_n,
    input  wire strobe,
    output reg  do_op,    // one-cycle execute pulse
    output wire done      // one-cycle completion pulse
);

  localparam S_WAIT_HIGH = 1'b0,   // arm: waiting for strobe to rise
             S_WAIT_LOW  = 1'b1;   // issued: waiting for strobe to fall (re-arm)

  reg state;
  reg [DONE_DELAY-1:0] done_sr;

  always @(posedge clk) begin
    if (!rst_n) begin
      state   <= S_WAIT_HIGH;
      do_op   <= 1'b0;
      done_sr <= {DONE_DELAY{1'b0}};
    end else begin
      do_op <= 1'b0;        // default one-shot
      case (state)
        S_WAIT_HIGH:
          if (strobe) begin
            do_op <= 1'b1;
            state <= S_WAIT_LOW;
          end
        S_WAIT_LOW:
          if (!strobe)
            state <= S_WAIT_HIGH;
      endcase
      // Delay line: shift the execute pulse to align `done` with commit.
      done_sr <= {done_sr[DONE_DELAY-2:0], do_op};
    end
  end

  assign done = done_sr[DONE_DELAY-1];

endmodule

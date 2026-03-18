#include <metal_stdlib>
using namespace metal;

#define MAX_BOARDSIZE 23

struct PartialPlacement {
    uint col;
    uint pos_diag;
    uint neg_diag;
    uint init_bitfield;  // pre-computed available positions for start_row
};

kernel void nqueen(
    device const PartialPlacement* partials [[buffer(0)]],
    device ulong*                  results  [[buffer(1)]],
    constant uint&                 board_size [[buffer(2)]],
    constant uint&                 start_row  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint stack[MAX_BOARDSIZE + 2];
    uint col_s[MAX_BOARDSIZE];
    uint pos_s[MAX_BOARDSIZE];
    uint neg_s[MAX_BOARDSIZE];

    const uint mask        = (1u << board_size) - 1u;
    const uint board_minus = board_size - 1u;
    uint numrows = start_row;

    col_s[numrows] = partials[tid].col;
    pos_s[numrows] = partials[tid].pos_diag;
    neg_s[numrows] = partials[tid].neg_diag;

    // sp=1: stack[0] is the sentinel slot; break when sp reaches 0
    uint sp = 1;
    uint bitfield = partials[tid].init_bitfield;
    ulong local_count = 0;

    for (;;) {
        // Check before computing lsb — avoids a wasted NEG+AND on every backtrack.
        if (bitfield == 0u) {
            bitfield = stack[--sp];
            if (sp == 0u) break;
            --numrows;
            continue;
        }

        uint lsb = uint(-int(bitfield)) & bitfield;
        bitfield &= ~lsb;

        if (numrows < board_minus - 1) {
            // Normal descent: push state and go one row deeper.
            uint n = numrows++;
            col_s[numrows] = col_s[n] | lsb;
            pos_s[numrows] = (pos_s[n] | lsb) << 1;
            neg_s[numrows] = (neg_s[n] | lsb) >> 1;
            stack[sp++] = bitfield;
            bitfield = mask & ~(col_s[numrows] | pos_s[numrows] | neg_s[numrows]);
        } else if (numrows == board_minus - 1) {
            // Second-to-last row: count all valid last-row placements in one
            // popcount instead of iterating through them one at a time.
            // Eliminates ~40-60% of total loop iterations.
            uint nc = col_s[numrows] | lsb;
            uint np = (pos_s[numrows] | lsb) << 1;
            uint nn = (neg_s[numrows] | lsb) >> 1;
            local_count += popcount(mask & ~(nc | np | nn));
        } else {
            // Last row (only reachable for tiny boards where start_row == board_minus).
            ++local_count;
        }
    }

    results[tid] = local_count;
}

#include <metal_stdlib>
using namespace metal;

#define MAX_BOARDSIZE 23

constant uint BOARD_SIZE [[function_constant(0)]];
constant uint START_ROW  [[function_constant(1)]];

struct PartialPlacement {
    uint col;
    uint pos_diag;
    uint neg_diag;
    uint init_bitfield;
};

/* Primary kernel: each thread solves one partial placement to completion.
   No checkpointing — K is raised high enough that threads complete
   within the GPU watchdog window. */
kernel void nqueen(
    device const PartialPlacement* partials [[buffer(0)]],
    device ulong*                  results  [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    struct Frame { uint bf; uint col; uint pos; uint neg; };
    Frame stk[MAX_BOARDSIZE + 2];

    const uint mask        = (1u << BOARD_SIZE) - 1u;
    const uint board_minus = BOARD_SIZE - 1u;
    uint numrows = START_ROW;

    uint cur_col = partials[tid].col;
    uint cur_pos = partials[tid].pos_diag;
    uint cur_neg = partials[tid].neg_diag;

    uint sp = 1;
    uint bitfield = partials[tid].init_bitfield;
    ulong local_count = 0;

    for (;;) {
        if (bitfield == 0u) {
            --sp;
            if (sp == 0u) break;
            bitfield = stk[sp].bf;
            cur_col  = stk[sp].col;
            cur_pos  = stk[sp].pos;
            cur_neg  = stk[sp].neg;
            --numrows;
            continue;
        }

        uint lsb = uint(-int(bitfield)) & bitfield;
        bitfield &= ~lsb;

        if (numrows < board_minus - 3) {
            stk[sp].bf  = bitfield;
            stk[sp].col = cur_col;
            stk[sp].pos = cur_pos;
            stk[sp].neg = cur_neg;
            sp++;
            cur_col = cur_col | lsb;
            cur_pos = (cur_pos | lsb) << 1;
            cur_neg = (cur_neg | lsb) >> 1;
            numrows++;
            bitfield = mask & ~(cur_col | cur_pos | cur_neg);
        } else if (numrows == board_minus - 3) {
            uint nc0 = cur_col | lsb;
            uint np0 = (cur_pos | lsb) << 1;
            uint nn0 = (cur_neg | lsb) >> 1;
            uint bf0 = mask & ~(nc0 | np0 | nn0);
            while (bf0) {
                uint lsb0 = uint(-int(bf0)) & bf0;
                bf0 &= ~lsb0;
                uint nc1 = nc0 | lsb0;
                uint np1 = (np0 | lsb0) << 1;
                uint nn1 = (nn0 | lsb0) >> 1;
                uint bf1 = mask & ~(nc1 | np1 | nn1);
                while (bf1) {
                    uint lsb1 = uint(-int(bf1)) & bf1;
                    bf1 &= ~lsb1;
                    local_count += popcount(mask & ~((nc1|lsb1)|((np1|lsb1)<<1)|((nn1|lsb1)>>1)));
                }
            }
        } else if (numrows == board_minus - 2) {
            uint nc1 = cur_col | lsb;
            uint np1 = (cur_pos | lsb) << 1;
            uint nn1 = (cur_neg | lsb) >> 1;
            uint bf1 = mask & ~(nc1 | np1 | nn1);
            while (bf1) {
                uint lsb1 = uint(-int(bf1)) & bf1;
                bf1 &= ~lsb1;
                local_count += popcount(mask & ~((nc1 | lsb1) | ((np1 | lsb1) << 1) | ((nn1 | lsb1) >> 1)));
            }
        } else if (numrows == board_minus - 1) {
            uint nc = cur_col | lsb;
            uint np = (cur_pos | lsb) << 1;
            uint nn = (cur_neg | lsb) >> 1;
            local_count += popcount(mask & ~(nc | np | nn));
        } else {
            ++local_count;
        }
    }

    results[tid] = local_count;
}

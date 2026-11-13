#ifndef NQ_PEEL_H
#define NQ_PEEL_H

/*
 * nq_peel.h - 32-bit N-Queens subtree counter with 4-row leaf peeling.
 *
 * A port of the loop in nq_metal.metal, written so the same source compiles
 * as a CUDA device function (nq_cuda.cu) and as a plain C function on the
 * host (test_peel.c), because the CUDA build cannot be tested on every
 * machine this repo is developed on.
 *
 * Compared with the plain DFS in solveTask()/solveTaskDevice():
 *   - masks are 32-bit (exact for boards up to 31 wide),
 *   - the working state lives in locals; the explicit stack is only touched
 *     for rows above the last five,
 *   - the last four rows run as nested loops with no stack traffic, and the
 *     final row is counted with a single popcount instead of one leaf at a time.
 *
 * Convention: counts every completion once (no symmetry weighting), the same
 * as the GPU contract in nq.c.
 *
 * Before including, a translation unit may define:
 *   NQ_PEEL_FN            function qualifiers (default: static inline)
 *   NQ_PEEL_POPCOUNT(x)   32-bit population count (default: __builtin_popcount)
 */

#include <stdint.h>

#ifndef NQ_PEEL_FN
#define NQ_PEEL_FN static inline
#endif
#ifndef NQ_PEEL_POPCOUNT
#define NQ_PEEL_POPCOUNT(x) __builtin_popcount(x)
#endif

#define NQ_PEEL_LSB(x) ((x) & (0u - (x)))

/* Largest board the 32-bit masks can represent. */
#define NQ_PEEL_MAX_BOARD 31

typedef struct
{
    uint32_t bf;
    uint32_t col;
    uint32_t pos;
    uint32_t neg;
} NqPeelFrame;

/*
 * Stack frames nqPeelSolve() can use when `rows` rows remain to be filled
 * (rows = board_size - numrows). A frame is pushed only while at least five
 * rows remain below the current one, so the depth is rows - 5. Never 0, so
 * callers can always allocate "this many" frames.
 */
#define NQ_PEEL_STACK_FRAMES(rows) ((rows) > 5 ? (rows) - 5 : 1)

/*
 * Count the completions of a partial placement.
 *
 *   mask         board_size low bits set
 *   board_minus  board_size - 1
 *   numrows      rows already filled = index of the row to fill next
 *   col/pos/neg  occupied columns and diagonals as seen from row `numrows`
 *   bitfield     free squares in row `numrows`
 *   stk          at least NQ_PEEL_STACK_FRAMES(board_size - numrows) frames
 */
NQ_PEEL_FN unsigned long long nqPeelSolve(uint32_t mask,
                                          uint32_t board_minus,
                                          uint32_t numrows,
                                          uint32_t col,
                                          uint32_t pos,
                                          uint32_t neg,
                                          uint32_t bitfield,
                                          NqPeelFrame* stk)
{
    unsigned long long count = 0;
    uint32_t sp = 0;

    for (;;)
    {
        uint32_t lsb;

        if (0u == bitfield)
        {
            if (0u == sp)
            {
                break;
            }
            --sp;
            bitfield = stk[sp].bf;
            col = stk[sp].col;
            pos = stk[sp].pos;
            neg = stk[sp].neg;
            --numrows;
            continue;
        }

        lsb = NQ_PEEL_LSB(bitfield);
        bitfield &= ~lsb;

        if (numrows + 5u <= board_minus)
        {
            /* At least five rows below this one: descend using the stack. */
            stk[sp].bf = bitfield;
            stk[sp].col = col;
            stk[sp].pos = pos;
            stk[sp].neg = neg;
            ++sp;
            col = col | lsb;
            pos = (pos | lsb) << 1;
            neg = (neg | lsb) >> 1;
            ++numrows;
            bitfield = mask & ~(col | pos | neg);
        }
        else if (numrows + 4u == board_minus)
        {
            /* Four rows below: rows N-4..N-1 in registers, last row by popcount. */
            uint32_t ca = col | lsb;
            uint32_t pa = (pos | lsb) << 1;
            uint32_t na = (neg | lsb) >> 1;
            uint32_t bfa = mask & ~(ca | pa | na);
            while (bfa)
            {
                uint32_t la = NQ_PEEL_LSB(bfa);
                uint32_t cb = ca | la;
                uint32_t pb = (pa | la) << 1;
                uint32_t nb = (na | la) >> 1;
                uint32_t bfb = mask & ~(cb | pb | nb);
                bfa &= ~la;
                while (bfb)
                {
                    uint32_t lb = NQ_PEEL_LSB(bfb);
                    uint32_t cc = cb | lb;
                    uint32_t pc = (pb | lb) << 1;
                    uint32_t nc = (nb | lb) >> 1;
                    uint32_t bfc = mask & ~(cc | pc | nc);
                    bfb &= ~lb;
                    while (bfc)
                    {
                        uint32_t lc = NQ_PEEL_LSB(bfc);
                        bfc &= ~lc;
                        count += (unsigned long long)NQ_PEEL_POPCOUNT(
                            mask & ~((cc | lc) | ((pc | lc) << 1) | ((nc | lc) >> 1)));
                    }
                }
            }
        }
        else if (numrows + 3u == board_minus)
        {
            uint32_t ca = col | lsb;
            uint32_t pa = (pos | lsb) << 1;
            uint32_t na = (neg | lsb) >> 1;
            uint32_t bfa = mask & ~(ca | pa | na);
            while (bfa)
            {
                uint32_t la = NQ_PEEL_LSB(bfa);
                uint32_t cb = ca | la;
                uint32_t pb = (pa | la) << 1;
                uint32_t nb = (na | la) >> 1;
                uint32_t bfb = mask & ~(cb | pb | nb);
                bfa &= ~la;
                while (bfb)
                {
                    uint32_t lb = NQ_PEEL_LSB(bfb);
                    bfb &= ~lb;
                    count += (unsigned long long)NQ_PEEL_POPCOUNT(
                        mask & ~((cb | lb) | ((pb | lb) << 1) | ((nb | lb) >> 1)));
                }
            }
        }
        else if (numrows + 2u == board_minus)
        {
            uint32_t ca = col | lsb;
            uint32_t pa = (pos | lsb) << 1;
            uint32_t na = (neg | lsb) >> 1;
            uint32_t bfa = mask & ~(ca | pa | na);
            while (bfa)
            {
                uint32_t la = NQ_PEEL_LSB(bfa);
                bfa &= ~la;
                count += (unsigned long long)NQ_PEEL_POPCOUNT(
                    mask & ~((ca | la) | ((pa | la) << 1) | ((na | la) >> 1)));
            }
        }
        else if (numrows + 1u == board_minus)
        {
            uint32_t ca = col | lsb;
            uint32_t pa = (pos | lsb) << 1;
            uint32_t na = (neg | lsb) >> 1;
            count += (unsigned long long)NQ_PEEL_POPCOUNT(mask & ~(ca | pa | na));
        }
        else
        {
            /* numrows == board_minus: this queen completes the board. */
            ++count;
        }
    }

    return count;
}

#endif

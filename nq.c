/*  Jeff Somers
 *
 *  Copyright (c) 2002
 *
 *  jsomers@alumni.williams.edu
 *  or
 *  allagash98@yahoo.com
 *
 *  April, 2002
 *  
 *  Program:  nq
 *  
 *  Program to find number of solutions to the N queens problem.
 *  This program assumes a twos complement architecture.
 *
 *  For example, you can arrange 4 queens on 4 x 4 chess so that
 *  none of the queens can attack each other:
 *
 *  Two solutions:
 *     _ Q _ _        _ _ Q _
 *     _ _ _ Q        Q _ _ _
 *     Q _ _ _        _ _ _ Q
 *     _ _ Q _    and _ Q _ _
 *    
 *  Note that these are separate solutions, even though they
 *  are mirror images of each other.
 *
 *  Likewise, a 8 x 8 chess board has 92 solutions to the 8 queens
 *  problem.
 *
 *  Command Line Usage:
 *
 *          nq N
 *
 *       where N is the size of the N x N board.  For example,
 *       nq 4 will find the 4 queen solution for the 4 x 4 chess
 *       board.
 *
 *  By default, this program will only print the number of solutions,
 *  not board arrangements which are the solutions.  To print the 
 *  boards, uncomment the call to printtable in the Nqueen function.
 *  Note that printing the board arrangements slows down the program
 *  quite a bit, unless you pipe the output to a text file:
 *
 *  nq 10 > output.txt
 *
 *
 *  The number of solutions for the N queens problems are known for
 *  boards up to 23 x 23.  With this program, I've calculated the
 *  results for boards up to 21 x 21, and that took over a week on
 *  an 800 MHz PC.  The algorithm is approximated O(n!) (i.e. slow), 
 *  and calculating the results for a 22 x 22 board will take about 8.5 
 *  times the amount of time for the 21 x 21 board, or over 8 1/2 weeks.
 *  Even with a 10 GHz machine, calculating the results for a 23 x 23 
 *  board would take over a month.  Of course, setting up a cluster of
 *  machines (or a distributed client) would do the work in less time.
 *  
 *  (from Sloane's On-Line Encyclopedia of Integer Sequences,
 *   Sequence A000170
 *   http://www.research.att.com/cgi-bin/access.cgi/as/njas/sequences/eisA.cgi?Anum=000170
 *   )
 *
 *   Board Size:       Number of Solutions to          Time to calculate 
 *   (length of one        N queens problem:              on 800MHz PC
 *    side of N x N                                    (Hours:Mins:Secs)
 *    chessboard)
 *
 *     1                                  1                    n/a
 *     2                                  0                   < 0 seconds
 *     3                                  0                   < 0 seconds
 *     4                                  2                   < 0 seconds
 *     5                                 10                   < 0 seconds
 *     6                                  4                   < 0 seconds
 *     7                                 40                   < 0 seconds
 *     8                                 92                   < 0 seconds
 *     9                                352                   < 0 seconds 
 *    10                                724                   < 0 seconds
 *    11                               2680                   < 0 seconds
 *    12                              14200                   < 0 seconds
 *    13                              73712                   < 0 seconds
 *    14                             365596                  00:00:01
 *    15                            2279184                  00:00:04
 *    16                           14772512                  00:00:23
 *    17                           95815104                  00:02:38
 *    18                          666090624                  00:19:26
 *    19                         4968057848                  02:31:24
 *    20                        39029188884                  20:35:06
 *    21                       314666222712                 174:53:45
 *    22                      2691008701644                     ?
 *    23                     24233937684440                     ?
 *    24                                  ?                     ?
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef WIN32
#include <pthread.h>
#include <unistd.h>
#endif

/*
   Notes on MAX_BOARDSIZE:

   This implementation uses 64-bit bitmasks and a 64-bit solution counter.
   MAX_BOARDSIZE is set to 27 so board sizes above 18 are supported while
   keeping the counter within practical 64-bit bounds for known counts.
*/

#ifdef WIN32
#define MAX_BOARDSIZE 27
typedef unsigned __int64 SOLUTIONTYPE;
typedef unsigned __int64 BITFIELDTYPE;
#define SOLUTION_PRINTF "%I64u"
#else
#define MAX_BOARDSIZE 27
typedef unsigned long long SOLUTIONTYPE;
typedef unsigned long long BITFIELDTYPE;
#define SOLUTION_PRINTF "%llu"
#endif

#define MIN_BOARDSIZE 2

SOLUTIONTYPE g_numsolutions = 0; 


/* Print a chess table with queens positioned for a solution */
/* This is not a critical path function & I didn't try to optimize it. */
void printtable(int boardsize, BITFIELDTYPE* aQueenBitRes, SOLUTIONTYPE numSolution)
{
    int i, j, k;
    BITFIELDTYPE row;

    /*  We only calculated half the solutions, because we can derive
        the other half by reflecting the solution across the "Y axis". */
    for (k = 0; k < 2; ++k)
    {
            printf("*** Solution #: " SOLUTION_PRINTF " ***\n", 2 * numSolution + k - 1);
        for ( i = 0; i < boardsize; i++)
        {
            BITFIELDTYPE bitf;
            /*
               Get the column that was set (i.e. find the
               first, least significant, bit set).
               If aQueenBitRes[i] = 011010b, then
               bitf = 000010b
            */
            bitf = aQueenBitRes[i];

            row = bitf ^ (bitf & (bitf - 1)); /* get least significant bit */
            for ( j = 0; j < boardsize; j++)
            {
                /* keep shifting row over to the right until we find the one '1' in
                   the binary representation.  There will only be one '1'. */
                if (0 == k && ((row >> j) & 1))
                {
                    printf("Q ");
                }
                else if (1 == k && (row & (((BITFIELDTYPE)1) << (boardsize - j - 1)))) /* this is the board reflected across the "Y axis" */
                {
                    printf("Q ");
                }
                else
                {
                    printf(". ");
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}

typedef struct
{
    int start_numrows;
    BITFIELDTYPE col;
    BITFIELDTYPE pos_diag;
    BITFIELDTYPE neg_diag;
    BITFIELDTYPE bitfield;
} NqueenTask;

#ifndef WIN32
typedef struct
{
    const NqueenTask* tasks;
    int task_count;
    int thread_id;
    int thread_count;
    int board_minus;
    BITFIELDTYPE mask;
    SOLUTIONTYPE subtotal;
    int started;
} WorkerContext;
#endif

static SOLUTIONTYPE solveTask(int board_minus, BITFIELDTYPE mask, const NqueenTask* task)
{
    BITFIELDTYPE aQueenBitCol[MAX_BOARDSIZE];
    BITFIELDTYPE aQueenBitPosDiag[MAX_BOARDSIZE];
    BITFIELDTYPE aQueenBitNegDiag[MAX_BOARDSIZE];
    BITFIELDTYPE aStack[MAX_BOARDSIZE + 2];
    BITFIELDTYPE bitfield;
    BITFIELDTYPE lsb;
    int numrows;
    BITFIELDTYPE* pnStack;
    SOLUTIONTYPE solutions = 0;

    aStack[0] = 0; /* stack sentinel */
    pnStack = aStack + 1;

    numrows = task->start_numrows;
    bitfield = task->bitfield;
    aQueenBitCol[numrows] = task->col;
    aQueenBitPosDiag[numrows] = task->pos_diag;
    aQueenBitNegDiag[numrows] = task->neg_diag;

    for (;;)
    {
        if (0 == bitfield)
        {
            bitfield = *--pnStack;
            if (pnStack == aStack)
            {
                break;
            }
            --numrows;
            continue;
        }

        /* get least-significant set bit */
        lsb = bitfield & (~bitfield + 1);
        bitfield &= ~lsb;

        if (numrows < board_minus)
        {
            int n = numrows++;
            aQueenBitCol[numrows] = aQueenBitCol[n] | lsb;
            aQueenBitNegDiag[numrows] = (aQueenBitNegDiag[n] | lsb) >> 1;
            aQueenBitPosDiag[numrows] = (aQueenBitPosDiag[n] | lsb) << 1;
            *pnStack++ = bitfield;
            bitfield = mask & ~(aQueenBitCol[numrows] | aQueenBitNegDiag[numrows] | aQueenBitPosDiag[numrows]);
            continue;
        }

        ++solutions;
        bitfield = *--pnStack;
        --numrows;
    }

    return solutions;
}

#ifndef WIN32
static int getThreadCount(int task_count)
{
    long cores;

    if (task_count <= 1)
    {
        return 1;
    }

    cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1)
    {
        cores = 1;
    }
    if (cores > task_count)
    {
        cores = task_count;
    }
    return (int)cores;
}

static void* NqueenWorker(void* arg)
{
    WorkerContext* context = (WorkerContext*)arg;
    int i;
    SOLUTIONTYPE subtotal = 0;

    for (i = context->thread_id; i < context->task_count; i += context->thread_count)
    {
        subtotal += solveTask(context->board_minus, context->mask, &context->tasks[i]);
    }

    context->subtotal = subtotal;
    return NULL;
}
#endif

/* The function which calculates the N queen solutions.
   We calculate one-half the solutions, then flip the results over
   the "Y axis" of the board.  Every solution can be reflected that
   way to generate another unique solution (assuming the board size
   isn't 1 x 1).  That's because a solution cannot be symmetrical
   across the Y-axis (because you can't have two queens in the same
   horizontal row).  A solution also cannot consist of queens 
   down the middle column of a board with an odd number of columns,
   since you can't have two queens in the same vertical row.

   This is a backtracking algorithm.  We place a queen in the top
   row, then note the column and diagonals it occupies.  We then 
   place a queen in the next row down, taking care not to place it
   in the same column or diagonal.  We then update the occupied
   columns & diagonals & move on to the next row.  If no position
   is open in the next row, we back track to the previous row & move
   the queen over to the next available spot in its row & the process
   starts over again.
*/
void Nqueen(int board_size)
{
    NqueenTask tasks[MAX_BOARDSIZE];
    int task_count = 0;
    BITFIELDTYPE bitfield;
    BITFIELDTYPE lsb;
    int i;
    int odd = board_size & 1; /* 0 if board_size even, 1 if odd */
    int half = board_size >> 1;
    int board_minus = board_size - 1; /* board size - 1 */
    BITFIELDTYPE mask = (((BITFIELDTYPE)1) << board_size) - 1; /* if board size is N, mask consists of N 1's */
    SOLUTIONTYPE total = 0;

    /* Build independent tasks for top-row placements (half board due symmetry). */
    bitfield = (((BITFIELDTYPE)1) << half) - 1;
    while (bitfield != 0)
    {
        NqueenTask* task = &tasks[task_count];

        lsb = bitfield & (~bitfield + 1);
        bitfield &= ~lsb;

        task->start_numrows = 1;
        task->col = lsb;
        task->neg_diag = (lsb >> 1);
        task->pos_diag = (lsb << 1);
        task->bitfield = mask & ~(task->col | task->neg_diag | task->pos_diag);
        ++task_count;
    }

    /* Odd board sizes need a second pass: first queen fixed at center column. */
    if (odd)
    {
        BITFIELDTYPE middle = ((BITFIELDTYPE)1) << half;
        bitfield = (middle - 1) >> 1;

        while (bitfield != 0)
        {
            BITFIELDTYPE col;
            BITFIELDTYPE neg_diag;
            BITFIELDTYPE pos_diag;
            NqueenTask* task = &tasks[task_count];

            lsb = bitfield & (~bitfield + 1);
            bitfield &= ~lsb;

            col = middle | lsb;
            neg_diag = ((middle >> 1) | lsb) >> 1;
            pos_diag = ((middle << 1) | lsb) << 1;

            task->start_numrows = 2;
            task->col = col;
            task->neg_diag = neg_diag;
            task->pos_diag = pos_diag;
            task->bitfield = mask & ~(col | neg_diag | pos_diag);
            ++task_count;
        }
    }

#ifndef WIN32
    {
        int thread_count = getThreadCount(task_count);

        if (thread_count > 1)
        {
            pthread_t threads[MAX_BOARDSIZE];
            WorkerContext contexts[MAX_BOARDSIZE];

            for (i = 0; i < thread_count; ++i)
            {
                contexts[i].tasks = tasks;
                contexts[i].task_count = task_count;
                contexts[i].thread_id = i;
                contexts[i].thread_count = thread_count;
                contexts[i].board_minus = board_minus;
                contexts[i].mask = mask;
                contexts[i].subtotal = 0;
                contexts[i].started = 0;

                if (0 == pthread_create(&threads[i], NULL, NqueenWorker, &contexts[i]))
                {
                    contexts[i].started = 1;
                }
                else
                {
                    /* If we can't create this thread, run its slice inline. */
                    NqueenWorker(&contexts[i]);
                }
            }

            for (i = 0; i < thread_count; ++i)
            {
                if (contexts[i].started)
                {
                    pthread_join(threads[i], NULL);
                }
                total += contexts[i].subtotal;
            }
        }
        else
        {
            for (i = 0; i < task_count; ++i)
            {
                total += solveTask(board_minus, mask, &tasks[i]);
            }
        }
    }
#else
    for (i = 0; i < task_count; ++i)
    {
        total += solveTask(board_minus, mask, &tasks[i]);
    }
#endif

    /* multiply solutions by two, to count mirror images */
    g_numsolutions = total * 2;
}

/* Print the results at the end of the run */
void printResults(time_t* pt1, time_t* pt2)
{
    double secs;
    int hours , mins, intsecs;

    printf("End: \t%s", ctime(pt2));
    secs = difftime(*pt2, *pt1);
    intsecs = (int)secs;
    printf("Calculations took %d second%s.\n", intsecs, (intsecs == 1 ? "" : "s"));

    /* Print hours, minutes, seconds */
    hours = intsecs/3600;
    intsecs -= hours * 3600;
    mins = intsecs/60;
    intsecs -= mins * 60;
    if (hours > 0 || mins > 0) 
    {
        printf("Equals ");
        if (hours > 0) 
        {
            printf("%d hour%s, ", hours, (hours == 1) ? "" : "s");
        }
        if (mins > 0)
        {           
            printf("%d minute%s and ", mins, (mins == 1) ? "" : "s");
        }
        printf("%d second%s.\n", intsecs, (intsecs == 1 ? "" : "s"));

    }
}

/* main routine for N Queens program.*/
int main(int argc, char** argv)
{
    time_t t1, t2;
    int boardsize;

    if (argc != 2) {
        printf("N Queens program by Jeff Somers.\n");
        printf("\tallagash98@yahoo.com or jsomers@alumni.williams.edu\n");
        printf("This program calculates the total number of solutions to the N Queens problem.\n");
        printf("Usage: nq <width of board>\n"); /* user must pass in size of board */
        return 0;
    }

    boardsize = atoi(argv[1]);

    /* check size of board is within correct range */
    if (MIN_BOARDSIZE > boardsize || MAX_BOARDSIZE < boardsize)
    {
        printf("Width of board must be between %d and %d, inclusive.\n", 
                                            MIN_BOARDSIZE, MAX_BOARDSIZE );
        return 0;
    }

    time(&t1);
    printf("N Queens program by Jeff Somers.\n");
    printf("\tallagash98@yahoo.com or jsomers@alumni.williams.edu\n");
    printf("Start: \t %s", ctime(&t1));

    Nqueen(boardsize); /* find solutions */
    time(&t2);

    printResults(&t1, &t2);

    if (g_numsolutions != 0)
    {
        printf("For board size %d, " SOLUTION_PRINTF " solution%s found.\n", boardsize, g_numsolutions, (g_numsolutions == 1 ? "" : "s"));
    }
    else
    {
        printf("No solutions found.\n");
    }

    return 0;
}

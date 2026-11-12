/*
 * nq_metal.m - Metal GPU backend for nq (stub; real implementation follows).
 */
#include "nq_types.h"

int solveTasksGpuMetal(int board_minus,
                       BITFIELDTYPE mask,
                       const NqueenTask* tasks,
                       int task_count,
                       SOLUTIONTYPE* total_out)
{
    (void)board_minus; (void)mask; (void)tasks; (void)task_count; (void)total_out;
    return 0; /* decline -> CPU */
}

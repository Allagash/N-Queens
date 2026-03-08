#ifndef NQ_TYPES_H
#define NQ_TYPES_H

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

typedef struct
{
    int start_numrows;
    BITFIELDTYPE col;
    BITFIELDTYPE pos_diag;
    BITFIELDTYPE neg_diag;
    BITFIELDTYPE bitfield;
} NqueenTask;

#endif

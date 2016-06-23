#ifndef STATS_H_
#define STATS_H_

#include <stdio.h>
#include <stdlib.h>

#define YES_SPLITMV
#define SAVE_HITCOUNT


#ifdef __cplusplus
extern "C" {
#endif


FILE *fp_STATS;

void init_STATS(void);
void add_STATS(double, int);



#ifdef __cplusplus
}
#endif


#endif

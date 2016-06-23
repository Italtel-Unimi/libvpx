
#include <stdio.h>
#include <stdlib.h>
#include "misc/STATS.h"


#ifdef __cplusplus
extern "C" {
#endif

void init_STATS(void) {


	fp_STATS = fopen("STATS.txt", "w");
	fprintf(fp_STATS, "[GPU] ME\t[CPU] EF\t[CPU] LF\n");

}



void add_STATS(double val, int new_line) {

	if (new_line)
		fprintf(fp_STATS, "%.2f\n", val);
	else
		fprintf(fp_STATS, "%.2f\t", val);
}



#ifdef __cplusplus
}
#endif

// test correctness and performance of pulse-finding code
//
// --perf: do 1e9 samples
// otherwise read samples from stdin

#include <stdio.h>
#include <string.h>

#include "pulse_find.h"

vector <FILE*> fout;

void PULSE_FIND::pulse_complete(int level, double pulse_count, long nsamples) {
    fprintf(fout[level], "%ld %f\n", nsamples, pulse_count);
}

int main(int argc, char** argv) {
    PULSE_FIND pf;
    bool perf=false;
    if (argc>1) {
        if (!strcmp(argv[1], "--perf")) {
            perf = true;
        } else {
            printf("pulse_test [--perf] [< samples]\n");
            exit(1);
        }
    }
    if (perf) {
        int nlevels = 16;
        pf.init(nlevels, true);
        for (int i=0; i<1000000000; i++) {
            pf.add_sample(1);
        }
    } else {
        int nlevels = 4;
        char buf[256];

        pf.init(nlevels, false);
        for (int i=0; i<nlevels; i++) {
            sprintf(buf, "out_%d", i);
            fout.push_back(fopen(buf, "w"));
        }
        while (fgets(buf, 256, stdin)) {
            pf.add_sample(atof(buf));
        }
    }
}

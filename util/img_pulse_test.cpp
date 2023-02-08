// test correctness and performance of pulse-finding code
//
// --perf: do 1e9 samples
//  otherwise read samples from stdin
// --ncpus N: run N copies in parallel

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "img_pulse.h"

#define STATS
    // whether to compute mean/stddev

#ifdef STATS
#include "window_stats.h"
#define WIN_SIZE    64
#endif

vector <FILE*> fout;
bool perf=false;

void PULSE_FIND::pulse_complete(int level, double value, size_t isample) {
#if 0
    fprintf(fout[level], "%ld %f\n", isample, value);
#endif

#ifdef STATS
    WINDOW_STATS &wstats = levels[level].window_stats;
    wstats.add_value(value);
#endif
}

void usage() {
    printf("pulse_test [--perf] [< samples]\n");
    exit(1);
}

int main(int argc, char** argv) {
    int ncpus=0;
    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--perf")) {
            perf = true;
        } else if (!strcmp(argv[i], "--ncpus")) {
            ncpus = atoi(argv[++i]);
        } else {
            usage();
        }
    }

    if (perf) {
        int nlevels = 16;
        PULSE_FIND pf(nlevels, WIN_SIZE, 0);
#ifdef STATS
        printf("stats: yes\n");
#else
        printf("stats: no\n");
#endif

#ifdef LEVELS01
        printf("levels 0/1: yes\n");
#else
        printf("levels 0/1: no\n");
#endif

        if (ncpus) {
            for (int i=0; i<ncpus; i++) {
                if (!fork()) {
                    for (int i=0; i<1000000000; i++) {
                        pf.add_sample(1);
                    }
                    exit(0);
                }
            }
            for (int i=0; i<ncpus; i++) {
                int pid;
                wait(&pid);
            }
        } else {
            for (int i=0; i<1000000000; i++) {
                pf.add_sample(1);
            }
        }
    } else {
        int nlevels = 4;
        char buf[256];

        PULSE_FIND pf(nlevels, WIN_SIZE, 0);
        for (int i=0; i<nlevels; i++) {
            sprintf(buf, "out_%d", i);
            fout.push_back(fopen(buf, "w"));
        }
        while (fgets(buf, 256, stdin)) {
            pf.add_sample(atof(buf));
        }
    }
}

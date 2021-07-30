#ifndef PULSE_FIND_H
#define PULSE_FIND_H

// struct PULSE_FIND:
// given a sequence of samples,
// compute pulses at time scales increasing by powers of 2,
// In each time scale, look at 90 deg phases.
// compute statistics for each time scale.
// When a pulse is complete, call pulse_complete() (app must define this)

// Notes:
// 1) We do this in a streaming fashion;
//      call add_sample() for each sample.
//      You don't have to keep samples in memory.
// 2) We do this efficiently
//      Instead of adding each sample to counters for all time scales,
//      we aggregate sums moving up the time scales.
//      So we do an average of 3 floating-point adds per sample,
//      regardless of the number of levels.

#include <stdio.h>
#include <stdlib.h>
#include <vector>

using std::vector;

#define DEBUG 0

// LEVEL represents one of the time scales.
// It maintains 4 pulses in progress (different phases)
// and the statistics of pulses completed.
//
// Note: the first two levels (single samples, and pairs of samples)
// are different; they don't have 4 phases.
// We handle them separately.
//
struct LEVEL {
    double count[4];
    int phase;      // next pulse to start

    LEVEL() {
        phase = 0;
        count[0] = count[1] = count[2] = count[3] = 0;
    }
        
    // an even pulse from the next lower level is complete.
    // Add it either to 0/2 or 1/3, one of which is now complete.
    // If it's 0 or 2, return true, in which case we want
    // to pass the completed pulse to the next higher level..
    // In either case, return the value of the completed pulse in pulse_count
    //
    bool add_pulse(double &pulse_count) {
        int phase2 = (phase+2)&3;

        count[phase] = pulse_count;
        count[phase2] += pulse_count;
        pulse_count = count[phase2];
        count[phase2] = 0;
        phase = (phase+1)&3;
        return phase&1?true:false;
        printf("add pulse: return\n");
    }
};

struct PULSE_FIND {
    long nsamples;  // how many samples processed so far
    int nlevels;    // not counting 0 and 1
    vector<LEVEL> levels;
    bool perf;      // if set, no pulse_complete() calls

    // application must define this
    //
    void pulse_complete (int, double, long);

    PULSE_FIND(int _nlevels, bool _perf) {
        perf = _perf;
        nlevels = _nlevels-2;
        if (nlevels < 1) {
            fprintf(stderr, "nlevels must be > 2\n");
            exit(1);
        }
        LEVEL level;
        for (int i=0; i<nlevels; i++) {
            levels.push_back(level);
        }
        nsamples = 0;
    }

    void add_sample(double x) {
        static double odd_sum=0, even_sum=0;    // level 1
        double pulse_count;

        if (nsamples&1) {
            // odd sample; pulse 0 is complete
            even_sum += x;
            pulse_count = even_sum;
            odd_sum = x;
        } else {
            // even sample; pulse 1 is complete
            odd_sum += x;
            pulse_count = odd_sum;
            even_sum = x;
        }
        if (nsamples == 0) {
            nsamples = 1;
            return;
        }
        
        // Do higher levels.
        // At start of loop,
        // pulse_count is count of completed pulse from lower level
        //
        for (int i=0; i<nlevels; i++) {
#if DEBUG
            printf("add_pulse: level %d value %f nsamples %ld\n",
                i,pulse_count, nsamples
            );
#endif
            bool keep_going = levels[i].add_pulse(pulse_count);
            if (!perf) {
                pulse_complete(i, pulse_count, nsamples);
            }
            if (!keep_going) {
                break;
            }
        }
        nsamples++;
    }
};

#endif

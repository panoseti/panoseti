#ifndef WINDOW_STATS_H
#define WINDOW_STATS_H

// Given a stream of values,
// compute their mean and stddev over windows of a given size and spacing
// For example, if size=100 and spacing=10,
// the first window is values 0..99, the second window is 10..109, and so on.

// Note: we could maintain the mean by adding new values
// and subtracting old ones, but this could accumulate round-off error.
// So maintain vector of values, and compute its mean as needed

#include <math.h>
#include <vector>
#include <numeric>

using std::vector;

struct WINDOW_STATS {
    vector<double> values;      // the current window
    int window_size;
    int window_spacing;
    int pos;                    // where to write next into "values"
    int count;                  // spacing counter
    bool ready;                 // have we completed a window yet?
    double mean, stddev;

    WINDOW_STATS(int _window_size, int _window_spacing) {
        window_size = _window_size;
        window_spacing = _window_spacing;
        values.resize(window_size);
        pos = 0;
        count = 0;
        ready = false;
    }

    // compute mean and stddev of current window
    //
    void compute_stats() {
        mean = accumulate(values.begin(), values.end(), 0)/window_size;
        double sum = 0;
        for (unsigned int i=0; i<window_size; i++) {
            double x = values[i]-mean;
            sum += x*x;
        }
        sum /= window_size;
        stddev = sqrt(sum);
    }

    // return true if we computed a new window
    //
    bool add_value(double x) {
        values[pos] = x;
        if (++pos == window_size) {
            pos = 0;
            if (!ready) {
                ready = true;
                count = window_spacing-1;
            }
        }
        if (!ready) return false;
        if (++count == window_spacing) {
            compute_stats();
            count = 0;
            return true;
        }
        return false;
    }
};

#endif

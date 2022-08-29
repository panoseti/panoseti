#ifndef WINDOW_STATS_H
#define WINDOW_STATS_H

// Given a stream of values,
// compute their mean and variance over sliding windows of a given size

#include <vector>
#include <math.h>

using std::vector;

// see
// https://nestedsoftware.com/2019/09/26/incremental-average-and-standard-deviation-with-sliding-window-470k.176143.html

struct WINDOW_STATS {
    int window_size;
    vector<double> values;      // the current window
    int pos;
    double var_by_n;            // variance times window size
    double mean;

    void init(int _window_size) {
        window_size = _window_size;
        values.resize(window_size, 0);
        pos = 0;
        var_by_n = 0;
        mean = 0;
    }

    // add a value to end of window, drop start of window
    //
    void add_value(double x) {
        double old = values[pos];
        values[pos] = x;
        if (++pos == window_size) {
            pos = 0;
        }
        double new_mean = mean + (x - old)/window_size;
        var_by_n += (x - old)*(x - new_mean + old - mean);
        mean = new_mean;
    }

    inline double stddev() {
        return sqrt(var_by_n/window_size);
    }
};

#endif

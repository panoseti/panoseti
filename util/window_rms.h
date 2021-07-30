#ifndef WINDOW_RMS_H
#define WINDOW_RMS_H

// given a stream of values,
// compute their mean and RMS over windows of a given size and spacing

#include <math.h>
#include <vector>
#include <numeric>

using std::vector;

struct WINDOW_RMS {
    vector<double> values;
    int window_size;
    int window_spacing;
    int pos;
    int count;
    bool ready;
    double mean, rms;

    WINDOW_RMS(int _window_size, int _window_spacing) {
        window_size = _window_size;
        window_spacing = _window_spacing;
        values.resize(window_size);
        pos = 0;
        count = 0;
        ready = false;
    }

    // compute mean and RMS
    // Note: we could maintain mean by adding new values
    // and subtracting old ones, but this could accumulate round-off error.
    //
    void compute_rms() {
        mean = accumulate(values.begin(), values.end(), 0)/window_size;
        double sum = 0;
        for (unsigned int i=0; i<window_size; i++) {
            double x = values[i]-mean;
            sum += x*x;
        }
        rms = sqrt(sum);
        ready = true;
    }

    void add_value(double x) {
        values[pos] = x;
        pos = (pos+1)%window_size;
        count++;
        if (count == window_spacing) {
            compute_rms();
            count = 0;
        }
    }

};

#endif

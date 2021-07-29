// pulse [options]
// find pulses in a file and output:
//      - pulses above RMS threshold
//      - optionally, a log of means and RMSs
// The above are output separately for each pulse duration
// The output files are written in a directory hierarchy:
//      out_dir/
//          filename/  (from input file)
//              pixel/      (0.255)
//                  pulse_i     (i=pulse duration level: 0,1,...)
//                  stats_i
//
// --file x         data file
// --module n       module number, default 0
// --pixel n        pixel (0..255)
// --nlevels n      number of duration octaves (default 16)
// --win_size n     RMS window is n times pulse duration
//                  default: 256
// --win_spacing n  recompute RMS every n pulse durations
//                  default: 64
// --thresh x       threshold is x times RMS
//                  default: 10
// --out_dir x      output directory
//                  default: pulse_out
// --stats          output history of mean and RMS for each pulse duration i
//

#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ph5.h"
#include "pulse_find.h"

void usage() {
    printf("options:\n"
        "   --file x            data file\n"
        "   --module n          module number\n"
        "   --pixel n           pixel (0..255)\n"
        "   --nlevels n         duration levels (default 16)\n"
        "   --win_size n        RMS window is n times pulse duration\n"
        "                       default: 256\n"
        "   --win_spacing n     RMS window is n times pulse duration\n"
        "                       default: 256\n"
        "   --thresh x          threshold is x times RMS\n"
        "                       default: 10\n"
        "   --out_dir x         output directory\n"
        "                       default: pulse_out\n"
        "   --stats             output history of mean and RMS for each pulse duration\n"
    );
    exit(1);
}

double thresh = 10.;
bool stats = false;
vector<FILE*> pulse_fout;
vector<FILE*> stats_fout;

// called when a pulse is complete
//
void PULSE_FIND::pulse_complete(int level, double value, long isample) {
    if (stats) {
        fprintf(stats_fout[level], "%ld %f\n", isample, value);
    }
    WINDOW_RMS &wrms = levels[level].window_rms;
    wrms.add_value(value);
    if (wrms.ready) {
        if (value > wrms.mean + thresh*wrms.rms) {
            fprintf(pulse_fout[level], "%ld %f\n", isample, value);
        }
    }
}

int main(int argc, char **argv) {
    const char* file = NULL;
    int win_size = 256, win_spacing=64;
    int pixel, module;
    const char* out_dir = "pulse_out";
    int i, nlevels = 16;
    int retval;

    for (i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--file")) {
            file = argv[++i];
        } else if (!strcmp(argv[i], "--pixel")) {
            pixel = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--module")) {
            module = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--nlevels")) {
            nlevels = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--win_size")) {
            win_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--win_spacing")) {
            win_spacing = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--thresh")) {
            thresh = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--out_dir")) {
            out_dir = argv[++i];
        } else if (!strcmp(argv[i], "--stats")) {
            stats = true;
        } else {
            usage();
        }
    }
    if (!file) {
        usage();
    }

    PH5 ph5;
    retval = ph5.open(file);
    if (retval) {
        fprintf(stderr, "can't open %s\n", file);
        exit(1);
    }

    // create output directory and open output files
    //
    const char* file_name;
    file_name = strrchr(file, '/');
    if (file_name) {
        file_name++;
    } else {
        file_name = file;
    }
    char buf[1024];
    sprintf(buf, "%s/%s", out_dir, file_name);
    mkdir(buf, 0771);
    sprintf(buf, "%s/%s/%d", out_dir, file_name, pixel);
    mkdir(buf, 0771);
    printf("writing results to %s\n", buf);
    for (i=0; i<nlevels; i++) {
        sprintf(buf, "%s/%s/%d/pulse_%d", out_dir, file_name, pixel, i);
        pulse_fout.push_back(fopen(buf, "w"));
        if (stats) {
            sprintf(buf, "%s/%s/%d/stats_%d", out_dir, file_name, pixel, i);
            stats_fout.push_back(fopen(buf, "w"));
        }
    }

    // scan the data file
    //
    PULSE_FIND pulse_find;
    for (int ifg=0; ; ifg++) {
        FRAME_GROUP fg;
        retval = ph5.get_frame_group(
            "/bit16IMGData/ModulePair_00254_00001/DATA", ifg, fg
        );
        if (retval) break;

        for (int iframe=0; i<fg.nframes; i++) {
            uint16_t* p = fg.get_frame(iframe, module);
            uint16_t val = p[pixel];
            pulse_find.add_sample((double)val);
        }
    }
}

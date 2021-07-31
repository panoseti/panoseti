// pulse [options]
// find pulses in a file and output:
//      - pulses above RMS threshold
//      - optionally, a log of all pulses
//      - optionally, a log of means and RMSs
// The above are output separately for each pulse duration
// The output files are written in a directory hierarchy:
//      out_dir/
//          filename/  (from input file)
//              pixel/      (0.255)
//                  pulse_i     (i=pulse duration level: 0,1,...)
//                  all_i
//                  stats_i
//
// --file x         data file
// --module n       module number, 0/1 default 0
// --pixel n        pixel (0..1023)
// --nlevels n      number of duration octaves (default 16)
// --win_size n     RMS window is n times pulse duration
//                  default: 256
// --win_spacing n  recompute RMS every n pulse durations
//                  default: 64
// --thresh x       threshold is x times RMS
//                  default: 10
// --out_dir x      output directory
//                  default: pulse_out
// --log_pulses     output all pulses
// --log_stats      output history of mean and RMS for each pulse duration
//

#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ph5.h"
#include "pulse_find.h"
#include "window_rms.h"

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
        "   --thresh x          threshold is mean + x times RMS\n"
        "                       default: 1\n"
        "   --out_dir x         output directory\n"
        "                       default: pulse_out\n"
        "   --log_pulses        output all pulses\n"
        "   --log_stats         output history of mean and RMS for each pulse duration\n"
    );
    exit(1);
}

double thresh = 1;
bool log_stats = true, log_pulses=true;
vector<FILE*> pulse_fout;
vector<FILE*> stats_fout;
vector<FILE*> all_fout;
vector<WINDOW_RMS> window_rms;

// called when a pulse is complete
//
void PULSE_FIND::pulse_complete(int level, double value, long isample) {
    //printf("pulse complete: level %d value %f sample %ld\n", level, value, isample);
    if (log_pulses) {
        fprintf(all_fout[level], "%ld,%f\n", isample, value);
    }

    WINDOW_RMS &wrms = window_rms[level];
    bool new_window = wrms.add_value(value);
    if (new_window && log_stats) {
        fprintf(stats_fout[level], "%ld,%f,%f\n", isample, wrms.mean, wrms.rms);
    }

    //printf("wrms: ready %d mean %f rms %f\n", wrms.ready, wrms.mean, wrms.rms);
    if (wrms.ready) {
        if (value > wrms.mean + thresh*wrms.rms) {
            fprintf(pulse_fout[level], "%ld,%f\n", isample, value);
        }
    }
}

int main(int argc, char **argv) {
    const char* file = "PANOSETI_DATA/PANOSETI_LICK_2021_07_15_08-36-14.h5";
    int win_size = 256, win_spacing=64;
    int pixel=0, module=0;
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
        } else if (!strcmp(argv[i], "--log_stats")) {
            log_stats = true;
        } else if (!strcmp(argv[i], "--log_pulses")) {
            log_pulses = true;
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

    // set up the stats 

    WINDOW_RMS wrms(win_size, win_spacing);
    for (i=0; i<nlevels; i++) {
        window_rms.push_back(wrms);
    }

    // create output directory
    //
    const char* file_name;
    file_name = strrchr(file, '/');
    if (file_name) {
        file_name++;
    } else {
        file_name = file;
    }
    char buf[1024];
    mkdir(out_dir, 0771);
    sprintf(buf, "%s/%s", out_dir, file_name);
    mkdir(buf, 0771);
    sprintf(buf, "%s/%s/%d", out_dir, file_name, pixel);
    mkdir(buf, 0771);
    printf("writing results to %s\n", buf);

    // open output files
    //
    for (i=0; i<nlevels; i++) {
        sprintf(buf, "%s/%s/%d/pulse_%d", out_dir, file_name, pixel, i);
        FILE *f = fopen(buf, "w");
        if (!f) {
            printf("can't open %s\n", buf);
            exit(1);
        }
        pulse_fout.push_back(f);
        if (log_stats) {
            sprintf(buf, "%s/%s/%d/stats_%d", out_dir, file_name, pixel, i);
            FILE *f = fopen(buf, "w");
            fprintf(f, "frame,mean,rms\n");
            stats_fout.push_back(f);
        }
        if (log_pulses) {
            sprintf(buf, "%s/%s/%d/all_%d", out_dir, file_name, pixel, i);
            FILE*f = fopen(buf, "w");
            fprintf(f, "frame,value\n");
            all_fout.push_back(f);
        }
    }

    // scan data file
    //
    PULSE_FIND pulse_find(nlevels, false);
    for (int ifs=0; ifs<99999; ifs++) {
        FRAME_SET fs;
        retval = ph5.get_frame_set(
            "/bit16IMGData/ModulePair_00254_00001/DATA", ifs, fs
        );
        if (retval) break;

        printf("got %d frame pairs\n", fs.nframe_pairs);

        for (int iframe=0; iframe<fs.nframe_pairs; iframe++) {
            uint16_t* p = fs.get_mframe(iframe, module);
            uint16_t val = p[pixel];
            //printf("val: %d\n", val);

            pulse_find.add_sample((double)val);
        }
        printf("done with frame set %d\n",ifs);
    }
}

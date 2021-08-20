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
//                  pulse_i     pulses above threshold
//                      (i=pulse duration level: 0,1,...)
//                  all_i       all pulsese
//                  mean_i      mean
//                  rms_i       RMS
//                  value_i     pixel value
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
// --log_value      output all pixel values
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

#define MAX_VAL 1000        // ignore values larger than this
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
        "   --log_value         output pixel values\n"
        "   --log_stats         output history of mean and RMS for each pulse duration\n"
    );
    exit(1);
}

double thresh = 1;
bool log_stats = true, log_pulses=true, log_value=true;
vector<FILE*> pulse_fout;
vector<FILE*> mean_fout;
vector<FILE*> rms_fout;
vector<FILE*> all_fout;
FILE* value_fout;
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
        fprintf(mean_fout[level], "%ld,%f\n", isample, wrms.mean);
        fprintf(rms_fout[level], "%ld,%f\n", isample, wrms.rms);
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
    char buf[1024], file_dir[1024];
    mkdir(out_dir, 0771);
    sprintf(buf, "%s/%s", out_dir, file_name);
    mkdir(buf, 0771);
    sprintf(buf, "%s/%s/%d", out_dir, file_name, module);
    mkdir(buf, 0771);
    sprintf(file_dir, "%s/%s/%d/%d", out_dir, file_name, module, pixel);
    mkdir(file_dir, 0771);
    printf("writing results to %s\n", file_dir);


    // open output files
    //
    for (i=0; i<nlevels; i++) {
        sprintf(buf, "%s/pulse_%d", file_dir, i);
        FILE *f = fopen(buf, "w");
        if (!f) {
            printf("can't open %s\n", buf);
            exit(1);
        }
        pulse_fout.push_back(f);
        if (log_stats) {
            sprintf(buf, "%s/mean_%d", file_dir, i);
            FILE *f = fopen(buf, "w");
            fprintf(f, "frame,mean\n");
            mean_fout.push_back(f);

            sprintf(buf, "%s/rms_%d", file_dir, i);
            f = fopen(buf, "w");
            fprintf(f, "frame,rms\n");
            rms_fout.push_back(f);
        }
        if (log_pulses) {
            sprintf(buf, "%s/all_%d", file_dir, i);
            FILE*f = fopen(buf, "w");
            fprintf(f, "frame,value\n");
            all_fout.push_back(f);
        }
        if (log_value) {
            sprintf(buf, "%s/value", file_dir);
            value_fout = fopen(buf, "w");
            fprintf(value_fout, "frame,value\n");
        }
    }

    // scan data file
    //
    PULSE_FIND pulse_find(nlevels, false);
    int isample = 0;
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
            if (val > MAX_VAL) {
                val = 0;
            }

            pulse_find.add_sample((double)val);
            if (log_value) {
                fprintf(value_fout, "%d,%d\n", isample, val);
            }
            isample++;
        }
        printf("done with frame set %d\n",ifs);
    }
}

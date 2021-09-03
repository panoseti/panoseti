// pulse [options]
// find pulses in a file and output:
//      - pulses above stddev threshold
//      - optionally, a log of all pulses
//      - optionally, a log of mean and stddev
// The above are output separately for each pulse duration
// The output files are written in a directory hierarchy:
//      out_dir/
//          filename/  (from input file)
//              module/     (0..1)
//                  pixel/      (0.255)
//                      thresh_i     pulses above threshold
//                          (i=pulse duration level: 0,1,...)
//                      all_i       all pulsese
//                      mean_i      mean
//                      stddev_i    stddev
//
// options:
//
// --file x         data file
// --module n       module number, 0/1 default 0
// --pixel n        pixel (0..1023)
// --nlevels n      number of duration octaves (default 16)
// --win_size n     stats window is n times pulse duration
//                  default: 64
// --win_spacing n  recompute stats every n pulse durations
//                  default: 16
// --thresh x       threshold is x times stddev
//                  default: 1
// --out_dir x      output directory
//                  default: pulse_out
// --log_pulses     output pulses length 4 and up
// --log_stats      output history of stats for each pulse duration
//

#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "ph5.h"
#include "pulse_find.h"
#include "window_stats.h"

#define WIN_SIZE_DEFAULT    64
#define WIN_SPACING_DEFAULT 16
#define MAX_VAL             2000        // ignore values larger than this

void usage() {
    printf("options:\n"
        "   --file x            data file\n"
        "   --module n          module number\n"
        "   --pixel n           pixel (0..255)\n"
        "   --nlevels n         duration levels (default 16)\n"
        "   --win_size n        stats window is n times pulse duration\n"
        "                       default: 64\n"
        "   --win_spacing n     stats window computed every n samples\n"
        "                       default: 16\n"
        "   --thresh x          threshold is mean + x times stddev\n"
        "                       default: 1\n"
        "   --out_dir x         output directory\n"
        "                       default: pulse_out\n"
        "   --log_pulses        output pulses length 4 and up\n"
        "   --log_stats         output history of mean and stddev for each pulse duration\n"
    );
    exit(1);
}

double thresh = 3;
bool log_stats = true, log_pulses=true;
vector<FILE*> thresh_fout;
vector<FILE*> mean_fout;
vector<FILE*> stddev_fout;
vector<FILE*> all_fout;
vector<WINDOW_STATS> window_stats;
int nlevels = 16;

inline double sample_to_sec(long i) {
    return ((double)i)/200.;
}

// called when a pulse is complete
//
void PULSE_FIND::pulse_complete(int level, double value, long isample) {
    //printf("pulse complete: level %d value %f sample %ld\n", level, value, isample);

    int idur = 1<<level;
    isample = isample + 1 - idur;
    if (isample < 0) return;
    value /= idur;

    WINDOW_STATS &wstats = window_stats[level];
    bool new_window = wstats.add_value(value);
    if (new_window && log_stats) {
        fprintf(mean_fout[level], "%f,%f\n", sample_to_sec(isample), wstats.mean);
        fprintf(stddev_fout[level], "%f,%f\n", sample_to_sec(isample), wstats.stddev);
    }

#if 0
    printf("pulse_complete: level %d value %f ready %d mean %f stddev %f\n",
        level, value, wstats.ready, wstats.mean, wstats.stddev
    );
#endif
    double nsigma = 0;
    if (wstats.ready) {
        if (value > wstats.mean) {
            nsigma = (value-wstats.mean)/wstats.stddev;
            if (nsigma > thresh) {
                fprintf(thresh_fout[level], "%f,%f\n", sample_to_sec(isample), value);
            }
        }
    }
    if (log_pulses) {
        fprintf(all_fout[level], "%f,%f,%f\n", sample_to_sec(isample), value, nsigma);
    }

}

// open output files
//
void open_output_files(const char* file_dir) {
    char buf[1024];
    for (int i=0; i<nlevels; i++) {
        sprintf(buf, "%s/thresh_%d", file_dir, i);
        FILE *f = fopen(buf, "w");
        if (!f) {
            printf("can't open %s\n", buf);
            exit(1);
        }
        fprintf(f, "frame,value\n");
        thresh_fout.push_back(f);
        if (log_stats) {
            sprintf(buf, "%s/mean_%d", file_dir, i);
            FILE *f = fopen(buf, "w");
            fprintf(f, "frame,mean\n");
            mean_fout.push_back(f);

            sprintf(buf, "%s/stddev_%d", file_dir, i);
            f = fopen(buf, "w");
            fprintf(f, "frame,stddev\n");
            stddev_fout.push_back(f);
        }
        if (log_pulses) {
            sprintf(buf, "%s/all_%d", file_dir, i);
            FILE*f = fopen(buf, "w");
            fprintf(f, "frame,value\n");
            all_fout.push_back(f);
        }
    }
}

// flush output files.
// This is because HDF5 crashes at random times :-(
//
void flush_output_files() {
    for (int i=0; i<nlevels; i++) {
        fflush(thresh_fout[i]);
        if (log_stats) {
            fflush(mean_fout[i]);
            fflush(stddev_fout[i]);
        }
        if (log_pulses) {
            fflush(all_fout[i]);
        }
    }
}

int main(int argc, char **argv) {
    const char* file = "PANOSETI_DATA/PANOSETI_LICK_2021_07_15_08-36-14.h5";
    int win_size = WIN_SIZE_DEFAULT, win_spacing=WIN_SPACING_DEFAULT;
    int pixel=0, module=0;
    const char* out_dir = "pulse_out";
    int i;
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

    WINDOW_STATS w(win_size, win_spacing);
    for (i=0; i<nlevels; i++) {
        window_stats.push_back(w);
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

    open_output_files(file_dir);

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
            isample++;
            flush_output_files();
        }
        printf("done with frame set %d\n",ifs);
    }
}

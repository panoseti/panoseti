// pulse --file path [options]
// find pulses in a file and output:
//      - pulses above stddev threshold
//      - optionally, a log of all pulses
//      - optionally, a log of mean and stddev
// The above are output separately for each pulse duration
//
// The output files are written in a directory determined as follows:
//      input filename is of the form D/F
//      out_dir/
//          D/
//              F/
//                  pixel/      (0..1023)
//
// In either case, the output files are:
//      thresh_i     pulses above threshold
//          (i=pulse duration level: 0,1,...)
//      all_i       all pulses
//      mean_i      mean
//      stddev_i    stddev
//
// options:
//
// --pixel n        pixel (0..1023)
// --nlevels n      number of duration octaves (default 16)
// --win_size n     stats window is n times pulse duration
//                  default: 64
// --thresh x       threshold is x times stddev
//                  default: 3
// --out_dir x      top-level output directory (see above)
//                  default: derived
// --log_pulses     output pulses
// --log_stats      output history of stats for each pulse duration
//

#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "pff.h"
#include "pulse_find.h"
#include "window_stats.h"

#define WIN_SIZE_DEFAULT    64
#define MAX_VAL             65536        // ignore values larger than this

int win_size = WIN_SIZE_DEFAULT;
int pixel=0;
const char* out_dir = "derived";

void usage() {
    printf("options:\n"
        "   --file x            data file\n"
        "   --pixel n           pixel (0..255)\n"
        "   --nlevels n         duration levels (default 16)\n"
        "   --win_size n        stats window is n times pulse duration\n"
        "                       default: 64\n"
        "   --thresh x          threshold is mean + x times stddev\n"
        "                       default: 1\n"
        "   --out_dir x         output directory\n"
        "                       default: derived\n"
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
    double stddev = wstats.stddev();
    if (log_stats) {
        fprintf(mean_fout[level], "%f,%f\n",
            sample_to_sec(isample), wstats.mean
        );
        fprintf(stddev_fout[level], "%f,%f\n",
            sample_to_sec(isample), stddev
        );
    }

#if 0
    printf("pulse_complete: level %d value %f mean %f stddev %f\n",
        level, value, wstats.mean, wstats.stddev
    );
#endif
    double nsigma = 0;
    if (value > wstats.mean && stddev>0) {
        nsigma = (value-wstats.mean)/stddev;
        if (nsigma > thresh) {
            fprintf(thresh_fout[level], "%f,%f\n",
                sample_to_sec(isample), value
            );
        }
    }
    if (log_pulses) {
        fprintf(all_fout[level], "%f,%f,%f\n",
            sample_to_sec(isample), value, nsigma
        );
    }

    // add this sample AFTER using the window stats
    //
    wstats.add_value(value);
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
// This is needed because HDF5 crashes at random times :-(
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

unsigned short image[1024];

int do_pff(const char* path) {
    string dir, file;
    int retval = pff_parse_path(path, dir, file);
    if (retval) {
        fprintf(stderr, "bad path: %s\n", path);
        exit(1);
    }
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "can't open %s\n", path);
        exit(1);
    }

    // create output directory
    //
    char buf[1024], file_dir[1024];
    mkdir(out_dir, 0775);
    sprintf(buf, "%s/%s", out_dir, dir.c_str());
    mkdir(buf, 0775);
    sprintf(buf, "%s/%s/%s", out_dir, dir.c_str(), file.c_str());
    mkdir(buf, 0775);
    sprintf(file_dir, "%s/%s/%s/%d", out_dir, dir.c_str(), file.c_str(), pixel);
    mkdir(file_dir, 0775);
    printf("writing results to %s\n", file_dir);
    open_output_files(file_dir);

    string s;
    PULSE_FIND pulse_find(nlevels);
    int isample = 0;
    while (1) {
        retval = pff_read_json(f, s);
        if (retval) break;
        retval = pff_read_image(f, sizeof(image), image);
        if (retval) break;
        uint16_t val = image[pixel];
        if (val >= MAX_VAL) {
            val = 0;
        }
        pulse_find.add_sample((double)val);
        isample++;
    }
}

int main(int argc, char **argv) {
    const char* file = 0;
    int i;
    int retval;

    for (i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--file")) {
            file = argv[++i];
        } else if (!strcmp(argv[i], "--pixel")) {
            pixel = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--nlevels")) {
            nlevels = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--win_size")) {
            win_size = atoi(argv[++i]);
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

    // set up the stats 

    WINDOW_STATS w(win_size);
    for (i=0; i<nlevels; i++) {
        window_stats.push_back(w);
    }

    if (ends_with(file, ".pff")) {
        do_pff(file);
    } else {
        fprintf(stderr, "unknown file type: %s\n", file);
        exit(1);
    }
}

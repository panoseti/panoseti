// pulse --infile path [options]
//
// find pulses in an image file (or 2 files) and output
//      (for each pulse duration level i: 0,1,...)
//
//      thresh_i     pulses above a stddev threshold
//          frame, value, cur_mean, cur_stddev, nsigma, pixel
//      all_i       all pulses (optional)
//          frame, value, cur_mean, cur_stddev, nsigma, pixel
//      The above are written in a fixed-length format
//      so you can seek to a given frame in all_i
//
// options:
//
// --infile2 path   2nd input file.  Take product of corresponding samples
//                  (not implemented)
// --pixel n        pixel (0..1023)
//                  default: do all pixels
// --nlevels n      number of duration octaves (default 16)
// --win_size n     stats window is n times pulse duration
//                  default: 64
// --thresh x       threshold is x times stddev
//                  default: 3
// --out_dir x      output directory
// --log_all        output all pulses
// --nframes n      do only first n frames
// --bits_per_pixel default: 16
//

#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "pff.h"
#include "img_pulse.h"

#define MAX_VAL16            65536        // ignore values larger than this
#define MAX_VAL8             256

int win_size = -1;
int pixel=-1;
double nsec = 0;
const char* out_dir = ".";
double thresh = -1;
bool log_all = false;
vector<FILE*> thresh_fout;
vector<FILE*> all_fout;
int nlevels = -1;
long nframes=0;
int bits_per_pixel = 16;

void usage() {
    printf("options:\n"
        "   --infile x          input file name\n"
        "   --infile2 x         2nd input file name\n"
        "   --pixel n           pixel, 0..1023 (default: all pixels)\n"
        "   --nlevels n         duration levels (default 16)\n"
        "   --win_size n        stats window is n times pulse duration\n"
        "                       default: 64\n"
        "   --thresh x          threshold is mean + x times stddev\n"
        "                       default: 1\n"
        "   --out_dir x         output directory\n"
        "                       default: .\n"
        "   --log_all           output all pulses length 4 and up\n"
        "   --nframes N         do only first N frames\n"
        "   --bits_per_pixel N  default: 16\n"
    );
    exit(1);
}

inline double sample_to_sec(long i) {
    return ((double)i)/200.;
}

// called when a pulse is complete
//
void PULSE_FIND::pulse_complete(int level, double value, size_t isample) {
    int idur = 1<<level;
    isample = isample + 1 - idur;
    if (isample < 0) return;
    value /= idur;

    WINDOW_STATS &wstats = levels[level].window_stats;
    double stddev = wstats.stddev();
#if 0
    printf("pulse_complete: level %d value %f mean %f stddev %f\n",
        level, value, wstats.mean, stddev
    );
#endif
    double nsigma = 0;
    if (value > wstats.mean && stddev>0) {
        nsigma = (value-wstats.mean)/stddev;
        if (nsigma > thresh) {
            fprintf(thresh_fout[level], "%10ld,%12.4e,%12.4e,%12.4e,%12.4e,%4d\n",
                isample, value, wstats.mean, stddev, nsigma, pixel
            );
        }
    }
    if (log_all) {
        fprintf(all_fout[level], "%10ld,%12.4e,%12.4e,%12.4e,%12.4e,%4d\n",
            isample, value, wstats.mean, stddev, nsigma, pixel
        );
    }

    // add this sample AFTER using the window stats
    //
    wstats.add_value(value);
}

void open_output_files() {
    char buf[1024];
    for (int i=0; i<nlevels; i++) {
        sprintf(buf, "%s/thresh_%d", out_dir, i);
        FILE *f = fopen(buf, "w");
        if (!f) {
            printf("can't open %s\n", buf);
            exit(1);
        }
        thresh_fout.push_back(f);

        if (log_all) {
            sprintf(buf, "%s/all_%d", out_dir, i);
            FILE*f = fopen(buf, "w");
            all_fout.push_back(f);
        }
    }
}

unsigned short image16[1024];
unsigned char image8[1024];

void do_pixel(const char* infile) {
    FILE *f = fopen(infile, "r");
    if (!f) {
        fprintf(stderr, "can't open %s\n", infile);
        exit(1);
    }
    open_output_files();

    string s;
    PULSE_FIND pulse_find(nlevels, win_size, pixel);
    int isample = 0;
    while (1) {
        int retval = pff_read_json(f, s);
        if (retval) break;
        double dval;
        if (bits_per_pixel == 16) {
            retval = pff_read_image(f, sizeof(image16), image16);
            if (retval) break;
            unsigned short val = image16[pixel];
            if (val >= MAX_VAL16) val = 0;
            dval = (double)val;
        } else {
            retval = pff_read_image(f, sizeof(image8), image8);
            if (retval) break;
            unsigned char val = image8[pixel];
            if (val >= MAX_VAL8) val = 0;
            dval = (double)val;
        }
        pulse_find.add_sample(dval);
        isample++;
        if (isample == nframes) break;
    }
}

void do_all_pixels(const char* infile) {
    FILE *f = fopen(infile, "r");
    if (!f) {
        fprintf(stderr, "can't open %s\n", infile);
        exit(1);
    }
    open_output_files();
    vector<PULSE_FIND*> pfs(1024);
    for (int i=0; i<1024; i++) {
        pfs[i] = new PULSE_FIND(nlevels, win_size, i);
    }

    int isample = 0;
    string s;
    while (1) {
        int retval = pff_read_json(f, s);
        if (retval) break;
        if (bits_per_pixel == 16) {
            retval = pff_read_image(f, sizeof(image16), image16);
        } else {
            retval = pff_read_image(f, sizeof(image8), image8);
        }
        if (retval) break;
        double dval;
        for (pixel=0; pixel<1024; pixel++) {
            if (bits_per_pixel == 16) {
                unsigned short val = image16[pixel];
                if (val >= MAX_VAL16) val = 0;
                dval = (double)val;
            } else {
                unsigned char val = image8[pixel];
                if (val >= MAX_VAL8) val = 0;
                dval = (double)val;
            }
            pfs[pixel]->add_sample(dval);
        }
        isample++;
        if (isample == nframes) break;
    }
}

int main(int argc, char **argv) {
    const char* infile = 0;
    int i;
    int retval;

    for (i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--infile")) {
            infile = argv[++i];
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
        } else if (!strcmp(argv[i], "--log_all")) {
            log_all = true;
        } else if (!strcmp(argv[i], "--nframes")) {
            nframes = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--bits_per_pixel")) {
            bits_per_pixel = atoi(argv[++i]);
        } else {
            printf("unrecognized arg %s\n", argv[i]);
            usage();
        }
    }
    if (!infile || nlevels<0 || thresh<0 || win_size<0) {
        usage();
    }
    if (bits_per_pixel!=8 && bits_per_pixel!=16) {
        fprintf(stderr, "bad bits_per_pixel %d\n", bits_per_pixel);
    }

    if (!is_pff_file(infile)) {
        fprintf(stderr, "%s is not a PFF file\n", infile);
        exit(1);
    }
    if (pixel >= 0) {
        do_pixel(infile);
    } else {
        if (log_all) {
            fprintf(stderr, "can't use --log_all with all pixels");
        }
        do_all_pixels(infile);
    }
}

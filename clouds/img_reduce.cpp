#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <omp.h>
#include <x86intrin.h>

#include "pff.h"

#define MAX_IN_FILES 100
#define IMG_SIZE 1024

uint16_t nfiles = 0;
uint64_t nframes = 0;
int bits_per_pixel = 16;


/*

// Returns the sum of all the 16-bit pixel counts in the 1024-pixel image img16.
// Assumes Intel SSE2 SIMD hardware support
uint32_t reduce_image16(uint16_t *img16) {
    uint32_t result = 0;
    __m128i sum_vec, tmp;
    sum_vec = _mm_setzero_si128();  // Accumulates 4 uint32_t pixel sums in a 128-bit register
    // Sum 4 pixels at a time
    for (uint16_t i = 0; i < (IMG_SIZE / 4) * 4; i += 4) {
        tmp = _mm_loadu_si128((__m128i*) (img16 + i));
        sum_vec = _mm_add_epi32(sum_vec, tmp);
    }
    int tmp_arr[4];
    _mm_storeu_si128((__m128i *) tmp_arr, sum_vec);
    for (uint16_t j = 0; j < 4; j++) result += tmp_arr[j];
    for (uint16_t k = (NUM_ELEMS / 4) * 4; k < NUM_ELEMS; k++) result += (vals[k] > 127) ? vals[k] : 0;

    return result;
}
*/

uint32_t reduce_image16_nonSIMD(uint16_t *img16) {
    uint32_t result = 0;
    for (uint16_t i = 0; i < IMG_SIZE; i++) {
        result += image16[i];
    }
    return result;
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


uint16_t image16[1024];
uint8_t image8[1024];

void do_file(const char* infile, ) {
    FILE *f = fopen(infile, "r");
    if (!f) {
        fprintf(stderr, "can't open %s\n", infile);
        exit(1);
    }
    //open_output_files();

    string s;
    while (1) {
        int retval = pff_read_json(f, s);
        if (retval) break;
        uint32_t sum_val = 0;
        if (bits_per_pixel == 16) {
            retval = pff_read_image(f, sizeof(image16), image16);
            if (retval) break;


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



void usage() {
    printf("options:\n"
        "   --infile x          input file name\n"
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


int main(int argc, char **argv) {
    const char* infiles[MAX_IN_FILES] = 0;
    int i;
    int retval;

    for (i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--infile")) {
            while (is_pff_file(argv[++i])) {
                infiles[nfiles++] = argv[i];
            }
            i--;
        } else if (!strcmp(argv[i], "--nframes")) {
            nframes = atof(argv[++i]);
        } else {
            printf("unrecognized arg %s\n", argv[i]);
            usage();
        }
    }
    if (!infiles) {
        usage();
    }
    if (bits_per_pixel!=8 && bits_per_pixel!=16) {
        fprintf(stderr, "bad bits_per_pixel %d\n", bits_per_pixel);
    }

    if (nfiles == 0) {
        fprintf(stderr, "No valid PFF files found\n", infiles);
        exit(1);
    } else {
        for (int i = 0; i < nfiles; i++) {
            do_file() // TODO
        }
    }
}

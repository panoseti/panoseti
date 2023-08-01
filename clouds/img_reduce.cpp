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
#define PIXELS_PER_IMAGE 1024

uint16_t nfiles = 0;
uint8_t bits_per_pixel = 16;
uint8_t bytes_per_image = (bits_per_pixel / 8) * PIXELS_PER_IMAGE;

typedef struct FILE_INFO {
    const char* fname;
    int64_t file_size;
    int64_t nframes;
};

void usage() {
    printf("options:\n"
        "   --infile_name x     input file name\n"
        "   --infile_nframes n  number of frames in the input file. Must follow each input file name\n"
        "   --bits_per_pixel N  default: 16\n"
    );
    exit(1);
}

/*

// Returns the sum of all the 16-bit pixel counts in the 1024-pixel image img16.
// Assumes Intel SSE2 SIMD hardware support
uint32_t reduce_image16(uint16_t *img16) {
    uint32_t result = 0;
    __m128i sum_vec, tmp;
    sum_vec = _mm_setzero_si128();  // Accumulates 4 uint32_t pixel sums in a 128-bit register
    // Sum 4 pixels at a time
    for (uint16_t i = 0; i < (PIXELS_PER_IMAGE / 4) * 4; i += 4) {
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
    for (uint16_t i = 0; i < PIXELS_PER_IMAGE; i++) {
        result += image16[i];
    }
    return result;
}

/*
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
*/

void free_img_info()

void get_img_info(const char **infiles, uint16_t nfiles, **FILE_INFO info_arr) {
    // Check file and compute frame size 
    FILE *f = fopen(infiles, "r");
    if (!f) {
        fprintf(stderr, "can't open %s\n", infiles);
        exit(1);
    }
    for (int i = 0; i < nfiles; i++) {
        FILE_INFO *ret = (FILE_INFO *) malloc(sizeof(FILE_INFO));
        if (ret == NULL) {
            fprintf(stderr, "Could not malloc enough space\n");
            exit(1);
        }
        int rv = pff_read_json(f, s);
        if (retval != 0) {
            fprintf(stderr, "Image read error [%d] in %s at frame index %d \n", retval, infiles, 0);
            exit(1);
        }
        int64_t header_size = ftell(f);
        if (header_size == -1) {
            fprintf(stderr, "ftell error [%d] in %s at frame index %d \n", retval, infiles, 0);
            exit(1);
        }
        int64_t frame_size = header_size + bytes_per_image + 1;
        int64_t file_size = f.seek(f, 0, SEEK_END);
        int64_t nframes = file_size / frame_size;

        ret->file_size = file_size;
        ret->nframes = nframes;
        FILE_INFO[i] = ret;
    }
    fclose(f);
}



void do_file(const char* infile, uint64_t nframes, uint32_t **data_ptr) {
    //open_output_files();
    uint32_t *reduced = (uint32_t *) malloc(nframes * sizeof(uint32_t));
    uint16_t num_threads;
    #pragma omp parallel
    {
        FILE *f = fopen(infile, "r");
        uint16_t image16[1024];
        uint8_t image8[1024];

        num_threads = omp_get_num_threads(); 
        uint16_t thread_id = omp_get_thread_num();
        uint16_t chunk_size = nframes / num_threads;
        string s;
        for (int i = chunk_size * thread_id; i < chunk_size * (thread_id + 1); i++) {
            int retval = pff_read_json(f, s);
            if (retval != 0) {
                fprintf(stderr, "Image read error [%d] in %s at frame index %d \n", retval, infile, i);
                break;
            }
            if (bits_per_pixel == 16) {
                retval = pff_read_image(f, sizeof(image16), image16);
                if (retval) break;
                reduced[i] = reduce_image16_nonSIMD(image16);   // potential race condition?
            }
        }
        fclose(f);
    }
    *data_ptr = reduced;
}



int main(int argc, char **argv) {
    const char* infiles[MAX_IN_FILES];
    const uint_64 nframes[MAX_IN_FILES];
    int i;
    int retval;

    for (i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--infile_name")) {
            infile = argv[++i];
            if (!is_pff_file(infile)) {
                fprintf(stderr, "%s is not a PFF file\n", infile);
                exit(1);
            }
            nfiles++;
        } else if (!strcmp(argv[i], "--bits_per_pixel")) {
            bits_per_pixel = atoi(argv[++i]);
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
            do_file(infiles[i], nframes[i])
        }
    }
}

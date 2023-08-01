#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <omp.h>
#include <x86intrin.h>

#include "../util/pff.h"

#define MAX_IN_FILES 1000
#define PIXELS_PER_IMAGE 1024

#define BAD_IMG 0xFFFFFFFF

uint16_t nfiles = 0;
uint8_t bits_per_pixel = 16;
uint8_t bytes_per_image = (bits_per_pixel / 8) * PIXELS_PER_IMAGE;

typedef struct FILE_INFO {
    const char* fname;
    int64_t file_size;
    int64_t frame_size;
    int64_t nframes;
    int32_t seqno;
};


void usage() {
    printf("options:\n"
        "   --infiles f0, f1, ...   at least one valid image mode pff file name\n"
        "   --bits_per_pixel N      default: 16\n"
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

uint32_t reduce_image16_nonSIMD(uint16_t *image16) {
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

void free_img_info(FILE_INFO **info_arr) {
    for (int i = 0; i < MAX_IN_FILES; i++) {
        if (info_arr[i] != NULL){
            free(info_arr[i]);
        }
    }
}

FILE_INFO* get_img_info(const char *fname, int32_t seqno) {
    // Check file and compute frame size 
    FILE *f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "can't open %s\n", fname);
        exit(1);
    }
    FILE_INFO *finfo = (FILE_INFO *) malloc(sizeof(FILE_INFO));
    if (finfo == NULL) {
        fprintf(stderr, "Could not malloc enough space\n");
        exit(1);
    }
    string s;
    int retval = pff_read_json(f, s);
    if (retval != 0) {
        fprintf(stderr, "Image read error [%d] in %s at frame index %d \n", retval, fname, 0);
        exit(1);
    }
    int64_t header_size = ftell(f);
    if (header_size == -1) {
        fprintf(stderr, "ftell error [%d] in %s at frame index %d \n", retval, fname, 0);
        exit(1);
    }
    int64_t frame_size = header_size + bytes_per_image + 1;
    fseek(f, 0, SEEK_END);
    int64_t file_size = ftell(f);
    int64_t nframes = file_size / frame_size;

    finfo->file_size = file_size;
    finfo->frame_size = frame_size;
    finfo->nframes = nframes;
    finfo->seqno = seqno;
    fclose(f);
    return finfo;
}


void do_file(const FILE_INFO *fin_info, FILE *fout_ptr) {
    //open_output_files();
    uint32_t reduced[fin_info->nframes];
    // uint32_t *reduced = (uint32_t *) malloc(nframes * sizeof(uint32_t));
    uint16_t num_threads, chunk_size;
    #pragma omp parallel
    {
        // Each thread processes a mutually exclusive set of image frames.
        FILE *fin_ptr = fopen(fin_info->fname, "r");
        uint16_t image16[1024];
        uint8_t image8[1024];

        num_threads = omp_get_num_threads();
        uint16_t thread_id = omp_get_thread_num();
        chunk_size = fin_info->nframes / num_threads;
        string s;
        // Last chunk may have a different amount of frames
        int64_t upper_limit = thread_id != (num_threads - 1) ? chunk_size * (thread_id + 1) : fin_info->nframes;
        for (int64_t i = chunk_size * thread_id; i < upper_limit; i++) {
            fseek(fin_ptr, chunk_size*thread_id*(fin_info->frame_size), SEEK_SET);
            int retval = pff_read_json(fin_ptr, s);
            if (retval != 0) {
                fprintf(stderr, "Image read error [%d] in %s at frame index %ld \n", retval, fin_info->fname, i);
                break;
            }
            if (bits_per_pixel == 16) {
                retval = pff_read_image(fin_ptr, sizeof(image16), image16);
                if (retval) {
                    reduced[i] = BAD_IMG;
                    continue;
                }
                reduced[i] = reduce_image16_nonSIMD(image16);   // potential race condition?
            }
        }
        fclose(fin_ptr);
    }
    // Write data to output file.
    fwrite(&reduced, sizeof(uint32_t), fin_info->nframes, fout_ptr);
}



int main(int argc, char **argv) {
    FILE_INFO *info_arr[MAX_IN_FILES];
    int i;
    int retval;

    for (i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--infiles")) {
            while (is_pff_file(argv[++i])) {
                info_arr[nfiles] = get_img_info(argv[i], nfiles);
                nfiles++;
            }
            i--;
            // if (!is_pff_file(argv[++i])) {
            //     fprintf(stderr, "%s is not a PFF file\n", argv[i]);
            //     exit(1);
            // }
        } else if (!strcmp(argv[i], "--bits_per_pixel")) {
            bits_per_pixel = atoi(argv[++i]);
        } else {
            printf("unrecognized arg %s\n", argv[i]);
            usage();
        }
    }
    if (nfiles == 0) {
        usage();
    }
    if (bits_per_pixel!=8 && bits_per_pixel!=16) {
        fprintf(stderr, "bad bits_per_pixel %d\n", bits_per_pixel);
        exit(1);
    }
    if (nfiles == 0) {
        fprintf(stderr, "No valid PFF files found\n");
        exit(1);
    }

    char fout_name[] = "img_reduce_intermediate";
    FILE *fout = fopen(fout_name, "wb");

    for (int i = 0; i < nfiles; i++) {
        do_file(info_arr[i], fout);
    }

    free_img_info(info_arr);
}

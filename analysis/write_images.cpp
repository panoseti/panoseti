// write_images --bytes_per_pixel N [--nframes N] < infile.pff > outfile
//
// read a PFF image file;
// remove the JSON, write a binary file with just images
// (for display purposes)

DEPRECATED - IMAGES NOW HAVE FIXED-LENGTH HEADERS SO WE DON'T NEED THIS

#include <stdio.h>
#include <string.h>
#include "pff.h"

unsigned short image16[1024];
unsigned char image8[1024];

void print_frame(int bytes_per_pixel) {
    for (int i=0; i<32; i++) {
        for (int j=0; j<32; j++) {
            if (bytes_per_pixel == 2) {
                printf("%d ", image16[i*32+j]);
            } else {
                printf("%d ", image8[i*32+j]);
            }
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int retval;
    uint64_t nframes = 0;
    int bytes_per_pixel = 0;

    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--nframes")) {
            nframes = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--bytes_per_pixel")) {
            bytes_per_pixel = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage");
            exit(1);
        }
    }

    if (!bytes_per_pixel) {
        fprintf(stderr, "usage");
        exit(1);
    }

    uint64_t iframe = 0;
    while (1) {
        string s;
        int retval = pff_read_json(stdin, s);
        if (retval) break;
        if (bytes_per_pixel == 2) {
            retval = pff_read_image(stdin, 2048, image16);
        } else {
            retval = pff_read_image(stdin, 1024, image8);
        }
        //print_frame(bytes_per_pixel);

        if (retval) break;
        if (bytes_per_pixel == 2) {
            fwrite(image16, 2, 1024, stdout);
        } else {
            fwrite(image8, 1, 1024, stdout);
        }
        iframe += 1;
        if (iframe == nframes) break;
    }
}

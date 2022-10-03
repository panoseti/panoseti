// write_images [--nframes N] < infile.pff > outfile
//
// read a PFF image file;
// remove the JSON, write a binary file with just images
// (for display purposes)

#include <stdio.h>
#include <string.h>
#include "pff.h"

unsigned short image[1024];

void print_frame() {
    for (int i=0; i<32; i++) {
        for (int j=0; j<32; j++) {
            printf("%d ", image[i*32+j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int retval;
    char buf[1024];
    uint64_t nframes = 0;

    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--nframes")) {
            nframes = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage");
            exit(1);
        }
    }

    uint64_t iframe = 0;
    while (1) {
        string s;
        int retval = pff_read_json(stdin, s);
        if (retval) break;
        retval = pff_read_image(stdin, 2048, image);
        //print_frame();

        if (retval) break;
        fwrite(image, 2, 1024, stdout);
        iframe += 1;
        if (iframe == nframes) break;
    }
}

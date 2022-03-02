// write_images < infile.pff > outfile
//
// read a PFF image file;
// remove the JSON, write a binary file with just images
// (for display purposes)

#include <stdio.h>
#include <string.h>
#include "pff.h"

unsigned short image[1024];

int main(int argc, char **argv) {
    int retval;
    char buf[1024];

    while (1) {
        string s;
        int retval = pff_read_json(stdin, s);
        if (retval) break;
        retval = pff_read_image(stdin, 2048, image);
        if (retval) break;
        fwrite(image, 2, 1024, stdout);
    }
}

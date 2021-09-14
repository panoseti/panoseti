// write_images --file path
//
// read PFF file, write binary images

#include <stdio.h>
#include <string.h>
#include "pff.h"

unsigned short image[1024];

int main(int argc, char **argv) {
    const char* path;
    FILE *fout;
    int retval;
    char buf[1024];

    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--file")) {
            path = argv[++i];
        }
    }

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "can't open %s\n", path);
        exit(1);
    }

    string dir, file;
    pff_parse_path(path, dir, file);

    sprintf(buf, "pulse_out/%s/%s/images.bin", dir.c_str(), file.c_str());
    fout = fopen(buf, "w");

    while (1) {
        string s;
        int retval = pff_read_json(f, s);
        if (retval) break;

        retval = pff_read_image(f, 2048, image);
        if (retval) break;
        fwrite(image, 2, 1024, fout);
    }
}

DEPRECATED
// write_images --file x
//
// read HDF5 file, write binary images to 2 other files

#include <stdio.h>
#include <string.h>
#include "ph5.h"

int main(int argc, char **argv) {
    const char* file;
    FILE *fout[2];
    int retval;
    char buf[1024];

    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--file")) {
            file = argv[++i];
        }
    }

    const char* file_name;
    file_name = strrchr(file, '/');
    if (file_name) {
        file_name++;
    } else {
        file_name = file;
    }

    PH5 ph5;
    retval = ph5.open(file);
    if (retval) {
        fprintf(stderr, "can't open %s\n", file);
        exit(1);
    }

    for (int i=0; i<2; i++) {
        sprintf(buf, "pulse_out/%s/%d/images.bin", file_name, i);
        fout[i] = fopen(buf, "w");
    }

    for (int ifs=0; ifs<99999; ifs++) {
        FRAME_SET fs;
        retval = ph5.get_frame_set(
            "/bit16IMGData/ModulePair_00254_00001/DATA", ifs, fs
        );
        if (retval) break;

        printf("got %d frame pairs\n", fs.nframe_pairs);

        for (int iframe=0; iframe<fs.nframe_pairs; iframe++) {
            for (int i=0; i<2; i++) {
                uint16_t* p = fs.get_mframe(iframe, i);
                fwrite(p, 2, 1024, fout[i]);
                fflush(fout[i]);
            }
        }
    }
}

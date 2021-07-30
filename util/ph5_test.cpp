#include <stdio.h>
#include <stdlib.h>

#include "ph5.h"

#define PATH "/disks/centurion/b/carolyn/b/home/ryanl/PANOSETI_DATA/PANOSETI_LICK_2021_07_15_05-04-15.h5"

void print_frames(int iframe, FRAME_SET &fs) {
    for (int module=0; module<2; module++) {
        uint16_t* p = fs.get_mframe(iframe, module);
        printf("frame %d module %d\n", iframe, module);
        for (int i=0; i<MFRAME_DIM; i++) {
            for (int j=0; j<MFRAME_DIM; j++) {
                printf("%d ", p[i*MFRAME_DIM+j]);
            }
            printf("\n");
        }
    }
}

int main(int, char**) {
    int retval;
    string s;
    PH5 file;

    retval = file.open(PATH);
    if (retval) {
        printf("no file %s\n", PATH);
        exit(1);
    }
    retval = file.get_attr("dateCreated", s);
    printf("date: %s\n", s.c_str());

    for (int ifg=0; ; ifg++) {
        FRAME_SET fs;
        retval = file.get_frame_set(
            "/bit16IMGData/ModulePair_00254_00001/DATA", ifg, fs
        );
        if (retval) break;
        printf("read frame set %d; %d frames\n", ifg, fs.nframe_pairs);
        for (int i=0; i<1; i++) {
            print_frames(i, fs);
        }
    }
    file.close();
}

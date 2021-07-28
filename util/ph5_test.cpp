#include <stdio.h>
#include <stdlib.h>

#include "ph5.h"

#define PATH "/disks/centurion/b/carolyn/b/home/ryanl/PANOSETI_DATA/PANOSETI_LICK_2021_07_15_05-04-15.h5"

void print_frames(int iframe, FRAME_GROUP &fg) {
    for (int module=0; module<2; module++) {
        for (int quabo=0; quabo<4; quabo++) {
            uint16_t* p = fg.get_frame(iframe, module, quabo);
            printf("frame %d module %d quabo %d\n", iframe, module, quabo);
            for (int i=0; i<8; i++) {
                for (int j=0; j<8; j++) {
                    printf("%d ", p[i*8+j]);
                }
                printf("\n");
            }
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
        FRAME_GROUP fg;
        retval = file.get_frame_group(
            "/bit16IMGData/ModulePair_00254_00001/DATA", ifg, fg
        );
        if (retval) break;
        printf("read frame group %d\n", ifg);
        for (int i=0; i<1; i++) {
            print_frames(i, fg);
        }
    }
    file.close();
}

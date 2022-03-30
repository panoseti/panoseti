#include <string.h>

#include "image.h"

// irot is clockwise 0/90/180/270
//
void copy_and_rotate_short(SHORT_ROW *from, SHORT_ROW *to, int irot) {
    switch(irot) {
    case 0:
        memcpy(to, from, DIM*2);
        break;
    case 1:
        for (int i=0; i<DIM; i++) {
            for (int j=0; j<DIM; j++) {
                to[j*2].x[DIM-1-i] = from[i].x[j];
            }
        }
        break;
    case 2:
        for (int i=0; i<DIM; i++) {
            for (int j=0; j<DIM; j++) {
                to[(DIM-1-i)*2].x[DIM-1-j] = from[i].x[j];
            }
        }
        break;
    case 3:
        for (int i=0; i<DIM; i++) {
            for (int j=0; j<DIM; j++) {
                to[(DIM-1-j)*2].x[i] = from[i].x[j];
            }
        }
    }
}

void copy_and_rotate_char(CHAR_ROW *from, CHAR_ROW *to, int irot) {
    switch(irot) {
    case 0:
        memcpy(to, from, DIM);
        break;
    case 1:
        for (int i=0; i<DIM; i++) {
            for (int j=0; j<DIM; j++) {
                to[j*2].x[DIM-1-i] = from[i].x[j];
            }
        }
        break;
    case 2:
        for (int i=0; i<DIM; i++) {
            for (int j=0; j<DIM; j++) {
                to[(DIM-1-i)*2].x[DIM-1-j] = from[i].x[j];
            }
        }
        break;
    case 3:
        for (int i=0; i<DIM; i++) {
            for (int j=0; j<DIM; j++) {
                to[(DIM-1-j)*2].x[i] = from[i].x[j];
            }
        }
    }
}

void image_combine_short(
    short *in0, short *in1, short *in2, short* in3, short *out
) {
    SHORT_ROW *p = (SHORT_ROW*) out;
    copy_and_rotate_short((SHORT_ROW*)in0, p, 0);
    copy_and_rotate_short((SHORT_ROW*)in1, p+1, 1);
    copy_and_rotate_short((SHORT_ROW*)in2, p+DIM*2-1, 2);
    copy_and_rotate_short((SHORT_ROW*)in3, p+DIM*2, 3);
}

void image_combine_char(
    char *in0, char *in1, char *in2, char* in3, char *out
) {
    CHAR_ROW *p = (CHAR_ROW*) out;
    copy_and_rotate_char((CHAR_ROW*)in0, p, 0);
    copy_and_rotate_char((CHAR_ROW*)in1, p+1, 1);
    copy_and_rotate_char((CHAR_ROW*)in2, p+DIM*2-1, 2);
    copy_and_rotate_char((CHAR_ROW*)in3, p+DIM*2, 3);
}

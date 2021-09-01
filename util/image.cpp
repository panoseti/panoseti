#include "image.h"

// functions to combine 4 16x16 images (rotated by 0/90/180/270)
// into an unrotated 32x32 image

struct SHORT_ROW {
    short x[DIM];
};

struct CHAR_ROW {
    char x[DIM];
};

// irot is clockwise 0/90/180/270
//
void rotate_short(SHORT_ROW *from, SHORT_ROW *to, int irot) {
    switch(irot) {
    case 0:
        for (int i=0; i<DIM; i++) {
            to[i*2] = from[i];
        }
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

void rotate_char(CHAR_ROW *from, CHAR_ROW *to, int irot) {
    switch(irot) {
    case 0:
        for (int i=0; i<DIM; i++) {
            to[i*2] = from[i];
        }
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
    rotate_short((SHORT_ROW*)in0, p, 0);
    rotate_short((SHORT_ROW*)in1, p+1, 1);
    rotate_short((SHORT_ROW*)in2, p+DIM*2-1, 2);
    rotate_short((SHORT_ROW*)in3, p+DIM*2, 3);
}

void image_combine_char(
    char *in0, char *in1, char *in2, char* in3, char *out
) {
    CHAR_ROW *p = (CHAR_ROW*) out;
    rotate_char((CHAR_ROW*)in0, p, 0);
    rotate_char((CHAR_ROW*)in1, p+1, 1);
    rotate_char((CHAR_ROW*)in2, p+DIM*2-1, 2);
    rotate_char((CHAR_ROW*)in3, p+DIM*2, 3);
}

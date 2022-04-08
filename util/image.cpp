#include <string.h>
#include <stdio.h>

#include "image.h"

// in a module:
// quabo 0 (upper left) is unrotated
// quabo 1 (upper right) is rotated 90 deg
// quabo 2 (lower left) is rotated 270 deg
// quabo 3 (lower right) is rotated 180 deg
//
// rotations are clockwise
// the following functions undo the rotation

void quabo_to_module_copy(
    QUABO_IMG_CHAR in, int iquabo, MODULE_IMG_SHORT out
) {
    switch(iquabo) {
    case 0:     // no rotation
        for (int i=0; i<QUABO_DIM; i++) {
            for (int j=0; j<QUABO_DIM; j++) {
                out[i][j] = in[i][j];
            }
        }
        break;
    case 1:     // rotate -90 (= 270)
        for (int i=0; i<QUABO_DIM; i++) {
            for (int j=0; j<QUABO_DIM; j++) {
                out[QUABO_DIM-j-1][QUABO_DIM+i] = in[i][j];
            }
        }
        break;
    case 2:     // rotate -270 (= 90)
        for (int i=0; i<QUABO_DIM; i++) {
            for (int j=0; j<QUABO_DIM; j++) {
                out[MODULE_DIM-j-1][i] = in[i][j];
            }
        }
        break;
    case 3:     // rotate -180 (= 180)
        for (int i=0; i<QUABO_DIM; i++) {
            for (int j=0; j<QUABO_DIM; j++) {
                out[MODULE_DIM-i-1][MODULE_DIM-j-1] = in[i][j];
            }
        }
        break;
    }
}

void quabo_to_module_add(
    QUABO_IMG_CHAR in, int iquabo, MODULE_IMG_SHORT out
) {
    switch(iquabo) {
    case 0:     // no rotation
        for (int i=0; i<QUABO_DIM; i++) {
            for (int j=0; j<QUABO_DIM; j++) {
                out[i][j] += in[i][j];
            }
        }
        break;
    case 1:     // rotate -90 (= 270)
        for (int i=0; i<QUABO_DIM; i++) {
            for (int j=0; j<QUABO_DIM; j++) {
                out[QUABO_DIM-j-1][QUABO_DIM+i] += in[i][j];
            }
        }
        break;
    case 2:     // rotate -270 (= 90)
        for (int i=0; i<QUABO_DIM; i++) {
            for (int j=0; j<QUABO_DIM; j++) {
                out[MODULE_DIM-j-1][i] += in[i][j];
            }
        }
        break;
    case 3:     // rotate -180 (= 180)
        for (int i=0; i<QUABO_DIM; i++) {
            for (int j=0; j<QUABO_DIM; j++) {
                out[MODULE_DIM-i-1][MODULE_DIM-j-1] += in[i][j];
            }
        }
        break;
    }
}

void print_quabo_img_char(QUABO_IMG_CHAR q) {
    for (int i=0; i<QUABO_DIM; i++) {
        for (int j=0; j<QUABO_DIM; j++) {
            printf("%3d ", q[i][j]);
        }
        printf("\n");
    }
}

void print_module_img_short(MODULE_IMG_SHORT m) {
    for (int i=0; i<MODULE_DIM; i++) {
        for (int j=0; j<MODULE_DIM; j++) {
            printf("%3d ", m[i][j]);
        }
        printf("\n");
    }
}

void zero_module_img_short(MODULE_IMG_SHORT m) {
    for (int i=0; i<MODULE_DIM; i++) {
        for (int j=0; j<MODULE_DIM; j++) {
            m[i][j] = 0;
        }
    }
}


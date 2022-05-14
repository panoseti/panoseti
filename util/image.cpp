#include <string.h>
#include <stdio.h>

#include "image.h"

// in a module:
// quabo 0 (upper left) is unrotated
// quabo 1 (upper right) is rotated 90 deg
// quabo 2 (lower right) is rotated 180 deg
// quabo 3 (lower left) is rotated 270 deg
//
// rotations are clockwise
// the following functions undo the rotation

// poor man's template

#define ROTATE_AND_OP \
    switch(iquabo) { \
    case 0: \
        for (int i=0; i<SRC_DIM; i++) { \
            for (int j=0; j<SRC_DIM; j++) { \
                (*out)[j][SRC_DIM-i-1] OP (*in)[i][j]; \
            } \
        } \
        break; \
    case 1: \
        for (int i=0; i<SRC_DIM; i++) { \
            for (int j=0; j<SRC_DIM; j++) { \
                (*out)[SRC_DIM-i-1][DST_DIM-j-1] OP (*in)[i][j]; \
            } \
        } \
        break; \
    case 2: \
        for (int i=0; i<SRC_DIM; i++) { \
            for (int j=0; j<SRC_DIM; j++) { \
                (*out)[DST_DIM-j-1][SRC_DIM+i] OP (*in)[i][j]; \
            } \
        } \
        break; \
    case 3: \
        for (int i=0; i<SRC_DIM; i++) { \
            for (int j=0; j<SRC_DIM; j++) { \
                (*out)[SRC_DIM+i][j] OP (*in)[i][j]; \
            } \
        } \
        break; \
    }

void quabo8_to_module8_copy(void *inp, int iquabo, void *outp) {
    QUABO_IMG8* in = (QUABO_IMG8*) inp;
    MODULE_IMG8* out = (MODULE_IMG8*) outp;
#define SRC_DIM QUABO_DIM
#define DST_DIM MODULE_DIM
#define OP =
    ROTATE_AND_OP
#undef SRC_DIM
#undef DST_DIM
#undef OP
}

void quabo8_to_module16_copy(void *inp, int iquabo, void *outp) {
    QUABO_IMG8* in = (QUABO_IMG8*) inp;
    MODULE_IMG16* out = (MODULE_IMG16*) outp;
#define SRC_DIM QUABO_DIM
#define DST_DIM MODULE_DIM
#define OP =
    ROTATE_AND_OP
#undef SRC_DIM
#undef DST_DIM
#undef OP
}

void quabo16_to_module16_copy(void *inp, int iquabo, void *outp) {
    QUABO_IMG16* in = (QUABO_IMG16*) inp;
    MODULE_IMG16* out = (MODULE_IMG16*) outp;
#define SRC_DIM QUABO_DIM
#define DST_DIM MODULE_DIM
#define OP =
    ROTATE_AND_OP
#undef SRC_DIM
#undef DST_DIM
#undef OP
}

void quabo16_to_quabo16_copy(void *inp, int iquabo, void *outp) {
    QUABO_IMG16* in = (QUABO_IMG16*) inp;
    QUABO_IMG16* out = (QUABO_IMG16*) outp;
#define SRC_DIM QUABO_DIM
#define DST_DIM QUABO_DIM
#define OP =
    ROTATE_AND_OP
#undef SRC_DIM
#undef DST_DIM
#undef OP
}

void quabo16_to_module16_add(void *inp, int iquabo, void *outp) {
    QUABO_IMG16* in = (QUABO_IMG16*) inp;
    MODULE_IMG16* out = (MODULE_IMG16*) outp;
#define SRC_DIM QUABO_DIM
#define DST_DIM MODULE_DIM
#define OP +=
    ROTATE_AND_OP
#undef SRC_DIM
#undef DST_DIM
#undef OP
}

void print_quabo_img8(QUABO_IMG8 q) {
    for (int i=0; i<QUABO_DIM; i++) {
        for (int j=0; j<QUABO_DIM; j++) {
            printf("%3d ", q[i][j]);
        }
        printf("\n");
    }
}

void print_module_img16(MODULE_IMG16 m) {
    for (int i=0; i<MODULE_DIM; i++) {
        for (int j=0; j<MODULE_DIM; j++) {
            printf("%3d ", m[i][j]);
        }
        printf("\n");
    }
}

void zero_module_img16(MODULE_IMG16 m) {
    for (int i=0; i<MODULE_DIM; i++) {
        for (int j=0; j<MODULE_DIM; j++) {
            m[i][j] = 0;
        }
    }
}


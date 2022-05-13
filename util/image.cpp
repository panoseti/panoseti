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

#define COPY \
    switch(iquabo) { \
    case 0: \
        for (int i=0; i<QUABO_DIM; i++) { \
            for (int j=0; j<QUABO_DIM; j++) { \
                (*out)[i][j] = (*in)[i][j]; \
            } \
        } \
        break; \
    case 1: \
        for (int i=0; i<QUABO_DIM; i++) { \
            for (int j=0; j<QUABO_DIM; j++) { \
                (*out)[j][MODULE_DIM-1-i] = (*in)[i][j]; \
            } \
        } \
        break; \
    case 2: \
        for (int i=0; i<QUABO_DIM; i++) { \
            for (int j=0; j<QUABO_DIM; j++) { \
                (*out)[MODULE_DIM-i-1][MODULE_DIM-j-1] = (*in)[i][j]; \
            } \
        } \
        break; \
    case 3: \
        for (int i=0; i<QUABO_DIM; i++) { \
            for (int j=0; j<QUABO_DIM; j++) { \
                (*out)[MODULE_DIM-j-1][i] = (*in)[i][j]; \
            } \
        } \
        break; \
    }

#define ADD \
    switch(iquabo) { \
    case 0: \
        for (int i=0; i<QUABO_DIM; i++) { \
            for (int j=0; j<QUABO_DIM; j++) { \
                (*out)[i][j] += (*in)[i][j]; \
            } \
        } \
        break; \
    case 1: \
        for (int i=0; i<QUABO_DIM; i++) { \
            for (int j=0; j<QUABO_DIM; j++) { \
                (*out)[j][MODULE_DIM-1-i] += (*in)[i][j]; \
            } \
        } \
        break; \
    case 2: \
        for (int i=0; i<QUABO_DIM; i++) { \
            for (int j=0; j<QUABO_DIM; j++) { \
                (*out)[MODULE_DIM-i-1][MODULE_DIM-j-1] += (*in)[i][j]; \
            } \
        } \
        break; \
    case 3: \
        for (int i=0; i<QUABO_DIM; i++) { \
            for (int j=0; j<QUABO_DIM; j++) { \
                (*out)[MODULE_DIM-j-1][i] += (*in)[i][j]; \
            } \
        } \
        break; \
    }

void quabo8_to_module8_copy(void *inp, int iquabo, void *outp) {
    QUABO_IMG8* in = (QUABO_IMG8*) inp;
    MODULE_IMG8* out = (MODULE_IMG8*) outp;
    COPY
}

void quabo8_to_module16_copy(void *inp, int iquabo, void *outp) {
    QUABO_IMG8* in = (QUABO_IMG8*) inp;
    MODULE_IMG16* out = (MODULE_IMG16*) outp;
    COPY
}

void quabo16_to_module16_copy(void *inp, int iquabo, void *outp) {
    QUABO_IMG16* in = (QUABO_IMG16*) inp;
    MODULE_IMG16* out = (MODULE_IMG16*) outp;
    COPY
}

void quabo16_to_module16_add(void *inp, int iquabo, void *outp) {
    QUABO_IMG16* in = (QUABO_IMG16*) inp;
    MODULE_IMG16* out = (MODULE_IMG16*) outp;
    ADD
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


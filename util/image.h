#ifndef PANOSETI_IMAGE_H_
#define PANOSETI_IMAGE_H_

// functions to copy and accumulate between quabo images (16x16)
// and module images (32x32)
// A module image is made up of 4 quabo images.
// The quabos are rotated by 0/90/180/270 on the mobo,
// so these rotations must be reversed.

#include <stdint.h>

#define QUABO_DIM 16
#define MODULE_DIM 32

typedef uint8_t QUABO_IMG_CHAR[QUABO_DIM][QUABO_DIM];
typedef uint16_t MODULE_IMG_SHORT[MODULE_DIM][MODULE_DIM];

extern void quabo_to_module_copy(
    QUABO_IMG_CHAR in, int iquabo, MODULE_IMG_SHORT out
);
extern void quabo_to_module_add(
    QUABO_IMG_CHAR in, int iquabo, MODULE_IMG_SHORT out
);

extern void print_quabo_img_char(QUABO_IMG_CHAR q);
extern void print_module_img_short(MODULE_IMG_SHORT m);
extern void zero_module_img_short(MODULE_IMG_SHORT m);

#endif

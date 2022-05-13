#ifndef PANOSETI_IMAGE_H_
#define PANOSETI_IMAGE_H_

// functions to copy and accumulate between quabo images (16x16)
// and module images (32x32)
// A module image is made up of 4 quabo images.
// The quabos are rotated by 0/90/180/270 on the mobo,
// so these rotations must be reversed.
//
// Images can be 8 or 16 bit

#include <stdint.h>

#define QUABO_DIM 16
#define MODULE_DIM 32

typedef uint8_t QUABO_IMG8[QUABO_DIM][QUABO_DIM];
typedef uint16_t QUABO_IMG16[QUABO_DIM][QUABO_DIM];
typedef uint8_t MODULE_IMG8[MODULE_DIM][MODULE_DIM];
typedef uint16_t MODULE_IMG16[MODULE_DIM][MODULE_DIM];

extern void quabo8_to_module8_copy(void* in, int iquabo, void* out);
extern void quabo8_to_module16_copy(void* in, int iquabo, void* out);
extern void quabo16_to_module16_copy(void* in, int iquabo, void* out);
extern void quabo16_to_quabo16_copy(void* in, int iquabo, void* out);
extern void quabo16_to_module16_add(void* in, int iquabo, void* out);

extern void print_quabo_img8(QUABO_IMG8 q);
extern void print_module_img16(MODULE_IMG16 m);
extern void zero_module_img16(MODULE_IMG16 m);

#endif

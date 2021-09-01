#ifndef PANOSETI_IMAGE_H_
#define PANOSETI_IMAGE_H_

// functions to combine 4 16x16 images (rotated by 0/90/180/270)
// into an unrotated 32x32 image

#define DIM 16

extern void image_combine_short(
    short *in0, short *in1, short *in2, short* in3, short *out
);

extern void image_combine_char(
    char *in0, char *in1, char *in2, char* in3, char *out
);

#endif

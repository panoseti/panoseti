#ifndef PANOSETI_IMAGE_H_
#define PANOSETI_IMAGE_H_

// functions to copy 16x16 images and rotate by 0/90/180/270

#define DIM 16

struct SHORT_ROW {
    short x[DIM];
};

struct CHAR_ROW {
    char x[DIM];
};

extern void copy_and_rotate_short(SHORT_ROW *from, SHORT_ROW *to, int irot);
extern void copy_and_rotate_shar(CHAR_ROW *from, CHAR_ROW *to, int irot);
extern void image_combine_short(
    short *in0, short *in1, short *in2, short* in3, short *out
);

extern void image_combine_char(
    char *in0, char *in1, char *in2, char* in3, char *out
);

#endif

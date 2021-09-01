#include <stdio.h>

#include "image.h"

short from_short[DIM*DIM], to_short[DIM*DIM*4];
char from_char[DIM*DIM], to_char[DIM*DIM*4];

int main(int, char**) {
    for (int i=0; i<DIM; i++) {
        for (int j=0; j<DIM; j++) {
            from_char[i*DIM+j] = i+j;
        }
    }
    for (int i=0; i<1000000; i++) {
        image_combine_char(from_char, from_char, from_char, from_char, to_char);
    }
    for (int i=0; i<DIM; i++) {
        for (int j=0; j<DIM; j++) {
            from_short[i*DIM+j] = i+j;
        }
    }
    for (int i=0; i<1000000; i++) {
        image_combine_short(from_short, from_short, from_short, from_short, to_short);
    }
    for (int i=0; i<DIM*2; i++) {
        for (int j=0; j<DIM*2; j++) {
            printf("%3d ", to_short[i*DIM*2 + j]);
        }
        printf("\n");
    }
}

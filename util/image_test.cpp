#include <stdio.h>

#include "image.h"

int main(int, char**) {
    QUABO_IMG_CHAR q;
    for (int i=0; i<QUABO_DIM; i++) {
        for (int j=0; j<QUABO_DIM; j++) {
            q[i][j] = i*QUABO_DIM +j;
        }
    }
    print_quabo_img_char(q);

    MODULE_IMG_SHORT m;
    zero_module_img_short(m);
    quabo_to_module_copy(q, 0, m);
    quabo_to_module_copy(q, 1, m);
    quabo_to_module_copy(q, 2, m);
    quabo_to_module_copy(q, 3, m);
    print_module_img_short(m);
}

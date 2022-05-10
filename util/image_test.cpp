#include <stdio.h>

#include "image.h"

int main(int, char**) {
    QUABO_IMG8 q;
    for (int i=0; i<QUABO_DIM; i++) {
        for (int j=0; j<QUABO_DIM; j++) {
            q[i][j] = i*QUABO_DIM +j;
        }
    }
    print_quabo_img8(q);
    printf("-----------\n");

    MODULE_IMG16 m;
    zero_module_img16(m);
    quabo8_to_module16_copy(q, 0, m);
    quabo8_to_module16_copy(q, 1, m);
    quabo8_to_module16_copy(q, 2, m);
    quabo8_to_module16_copy(q, 3, m);
    print_module_img16(m);
}

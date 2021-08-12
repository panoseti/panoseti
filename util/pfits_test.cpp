// example/test program for FITS API
//
// This writes a fits file consisting of
// 1) a header block with a struct FOO as key/value pairs
// 2) a 4x4 16-bit image, with a key/value pair

#include <stdio.h>
#include "pfits.h"

// The following shows how to convert between C structure and FITS header.
// NOTE: cfitsio converts all names into uppercase (huh??).
// So you must use uppercase names when parsing.
//
struct FOO {
    double x;
    long y;
    char blah[36];
    FOO() {
        x = 0;
        y = 0;
        strcpy(blah, "");
    }

    int pfits_write(PFITS &pf) {
        pf.put_double("X", x);
        pf.put_long("Y", y);
        pf.put_str("BLAH", blah);
        return 0;
    }
    int pfits_parse(PFITS &pf) {
        pf.get_double("X", x);
        pf.get_long("Y", y);
        pf.get_str("BLAH", blah);
        return 0;
    }
};

void write_file(const char* name) {
    FOO foo;
    foo.x = 3.14;
    foo.y = 17;
    strcpy(foo.blah, "foobar");
    short data[16] = {1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1};

    PFITS pf;

    pf.create_file(name);
    pf.create_header();
    foo.pfits_write(pf);

    pf.create_image(PF_U16, 4);
    pf.put_long("foobar", 1);
    pf.write_image(PF_U16, 4, data);
    pf.close();
}

void read_file(const char* name) {
    PFITS pf;
    FOO foo;
    short data[16];

    pf.open(name);
    pf.read_header();
    foo.pfits_parse(pf);
    printf("header: %f %ld %s\n", foo.x, foo.y, foo.blah);

    pf.next();
    pf.read_header();
    long foobar;
    pf.get_long("FOOBAR", foobar);
    printf("image header: %ld\n", foobar);
    pf.read_image(PF_U16, 4, data);
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            printf("%d ", data[i*4+j]);
        }
        printf("\n");
    }

}

int main(int, char**) {
    write_file("pfits.fits");
    read_file("pfits.fits");
}

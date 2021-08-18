// example/test program for FITS API
//
// pfits_test [--read | --write] [--nframes N]
//
// This writes a FITS file consisting of
// 1) a header block with a struct FOO as key/value pairs
// 2) N 32x32 16-bit images, with a key/value pair

#include <stdio.h>
#include "pfits.h"

#define IMAGE_DIM   32

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

void write_file(const char* name, int nframes) {
    FOO foo;
    foo.x = 3.14;
    foo.y = 17;
    strcpy(foo.blah, "foobar");
    short data[IMAGE_DIM][IMAGE_DIM];

    for (int i=0; i<IMAGE_DIM; i++) {
        for (int j=0; j<IMAGE_DIM; j++) {
            data[i][j] = i+j;
        }
    }

    PFITS pf;

    pf.create_file(name);
    pf.create_header();
    foo.pfits_write(pf);

    for (int i=0; i<nframes; i++) {
        pf.create_image(PF_16, IMAGE_DIM);
        pf.put_long("foobar", 1);
        pf.write_image(PF_16, IMAGE_DIM, data);
    }
    pf.close();
}

void read_file(const char* name) {
    PFITS pf;
    FOO foo;
    short data[IMAGE_DIM][IMAGE_DIM];
    int retval;

    pf.open(name);
    pf.read_header();
    foo.pfits_parse(pf);
    printf("header: %f %ld %s\n", foo.x, foo.y, foo.blah);
    retval = pf.next();

    for (int frame=0; ; frame++) {
        pf.read_header();
        long foobar;
        pf.get_long("FOOBAR", foobar);
        printf("frame %d\nheader: %ld\n", frame, foobar);
        pf.read_image(PF_16, IMAGE_DIM, data);
        for (int i=0; i<IMAGE_DIM; i++) {
            for (int j=0; j<IMAGE_DIM; j++) {
                printf("%2d ", data[i][j]);
            }
            printf("\n");
        }
        retval = pf.next();
        if (retval) break;
    }
    pf.close();
}

int main(int argc, char** argv) {
    bool do_read = false, do_write = false;
    int nframes = 1;
    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--read")) {
            do_read = true;
        } else if (!strcmp(argv[i], "--write")) {
            do_write = true;
        } else if (!strcmp(argv[i], "--nframes")) {
            nframes = atoi(argv[++i]);
        }
    }
    if (do_write) {
        write_file("pfits.fit", nframes);
    }
    if (do_read) {
        read_file("pfits.fit");
    }
}

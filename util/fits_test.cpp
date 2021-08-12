// FITS test
//
// Note: FITS is an antique.
// 80-col "cards"?  1-offset arrays?  gimme a break.
// Also the docs are weak.
//
// FITS cookbook:
// https://heasarc.gsfc.nasa.gov/docs/software/fitsio/cexamples/cookbook.c

#include "fitsio.h"

void check(int status, const char* where) {
    if (status) {
        printf("error in %s:\n", where);
        fits_report_error(stdout, status);
        exit(1);
    }
}

void write_file(const char* name) {
    fitsfile *f;
    long dim[2] = {4,4};
    long fpixel[2]= {1,1};
    int status = 0;
    short data[16] = {1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1};
    long x=14;

    fits_create_file(&f, name, &status);
    check(status, "create_file");

    // write initial header
    //
    fits_create_img(f, SHORT_IMG, 0, dim, &status);
    check(status, "create_img");
    for (int i=0; i<40; i++) {
        char buf[256];
        sprintf(buf, "HDR_%d", i);
        fits_update_key(f, TLONG, buf, &x, NULL, &status);
        check(status, "update_key");
    }
    fits_update_key(f, TLONG, "MAIN_HDR", &x, "main header", &status);
    check(status, "update_key");
    fits_update_key(f, TSTRING, "MAIN_H2", (void*)"foobar", "another header", &status);
    check(status, "update_key");

    // write image with header
    //
    fits_create_img(f, SHORT_IMG, 2, dim, &status);
    check(status, "create_img");

    fits_update_key(f, TLONG, "EXPOSURE", &x, "exposure time", &status);
    check(status, "update_key");
    fits_write_pix(f, TUSHORT, fpixel, 16, data, &status);
    check(status, "write_pix");
    fits_close_file(f, &status);
    check(status, "close_file");
}

void show_header(fitsfile *f, int i) {
    int nkeys, keypos;
    int status=0;

    printf("header %d\n", i);
    fits_get_hdrpos(f, &nkeys, &keypos, &status);
    char line[FLEN_CARD];
    for (int i=1; i<= nkeys; i++) {
        fits_read_record(f, i, line, &status);
        check(status, "read_record");
        printf("key: %s\n", line);
    }
}

void read_file(const char* name) {
    fitsfile *f;
    int bitpix;
    int naxis;
    long dim[2];
    int status=0;
    short data[16];
    long fpixel[2]= {1,1};
    short nulval=0;

    fits_open_file(&f, name, READONLY, &status);
    check(status, "open_file");
    show_header(f, 0);

    fits_movrel_hdu(f, 1, NULL, &status);
    check(status, "movrel_hdu");
    show_header(f, 1);

    fits_get_img_param(f, 2, &bitpix, &naxis, dim, &status);
    check(status, "get_img_params");
    printf("bitpix %d ndim %d dims %ld %ld\n", bitpix, naxis, dim[0], dim[1]);

    fits_read_pix(f, TUSHORT, fpixel, 16, NULL, data, NULL, &status);
    check(status, "read_pix");
    for (int i=0; i<16; i++) {
        printf("%d\n", data[i]);
    }
}

int main(int, char**) {
    //write_file("!foo.fits");
    read_file("pfits.fits");
}

#ifndef PANOSETI_FITS_H
#define PANOSETI_FITS_H

// API for reading and writing FITS files
//
// This API encapsulates cfitsio;
// you shouldn't have to call anything from it.
// See pfits_test.cpp for example usage.

#include <vector>
#include <string.h>
#include "fitsio.h"

using std::vector;

// our codes for numeric types
//
#define PF_8   0
#define PF_16  1
#define PF_32  2

// corresponding FITS codes
//
int fits_image_type[3] = {BYTE_IMG, SHORT_IMG, LONG_IMG};
int fits_data_type[3] = {TBYTE, TSHORT, TLONG};
int nbits[3] = {8, 16, 32};
int nbytes[3] = {1, 2, 4};

// a name/value pair from a FITS header
//
struct NAME_VAL {
    char name[9];
    char value[80];
};

struct PFITS {
    vector<NAME_VAL> name_vals;
    fitsfile *f;

    inline int check(int status, const char* where) {
        if (status) {
            printf("FITS error in %s:\n", where);
            fits_report_error(stdout, status);
            exit(1);
        }
    }
    int create_file(const char* name) {
        int status=0;
        char buf[256];
        sprintf(buf, "!%s", name);
        check(
            fits_create_file(&f, buf, &status),
            "create_file"
        );
        return status;
    }
    int open(const char* name) {
        int status=0;
        check(
            fits_open_file(&f, name, READONLY, &status),
            "open_file"
        );
    }
    int close() {
        int status=0;
        check(
            fits_close_file(f, &status),
            "close_file"
        );
        return status;
    }

    // stuff related to headers
    //
    int create_header() {
        int status=0;
        long dim = 0;
        check(
            fits_create_img(f, SHORT_IMG, 0, &dim, &status),
            "create_img"
        );
        return status;
    }
    int read_header() {
        int nkeys, keypos;
        int status=0;
        check(
            fits_get_hdrpos(f, &nkeys, &keypos, &status),
            "get_hdrpos"
        );
        if (status) return status;
        char line[FLEN_CARD];
        for (int i=1; i<= nkeys; i++) {
            fits_read_record(f, i, line, &status);
            char *p = strchr(line, '=');
            if (!p) continue;
            *p = 0;
            NAME_VAL nv;
            strcpy(nv.name, line);
            char *q = strchr(nv.name, ' ');   // strip trailing spaces
            if (q) *q = 0;
            strcpy(nv.value, p+1);
            name_vals.push_back(nv);
        }
        return 0;
    }
    bool get_double(const char* name, double& val) {
        for (unsigned int i=0; i<name_vals.size(); i++) {
            NAME_VAL& nv = name_vals[i];
            if (!strcmp(name, nv.name)) {
                sscanf(nv.value, "%lf", &val);
                return true;
            }
        }
        return false;
    }
    bool get_long(const char* name, long& val) {
        for (unsigned int i=0; i<name_vals.size(); i++) {
            NAME_VAL& nv = name_vals[i];
            if (!strcmp(name, nv.name)) {
                sscanf(nv.value, "%ld", &val);
                return true;
            }
        }
        return false;
    }
    bool get_str(const char* name, char* val) {
        for (unsigned int i=0; i<name_vals.size(); i++) {
            NAME_VAL& nv = name_vals[i];
            if (!strcmp(name, nv.name)) {
                char* p = strchr(nv.value, '\'');
                if (!p) return false;
                char *q = strchr(p+1, '\'');
                if (!q) return false;
                *q = 0;
                strcpy(val, p+1);
                return true;
            }
        }
        return false;
    }
    int put_long( const char* name, long val) {
        int status = 0;
        check(
            fits_update_key(f, TLONG, name, &val, NULL, &status),
            "update_key"
        );
        return status;
    }
    int put_double(const char* name, double val) {
        int status = 0;
        check(
            fits_update_key(f, TDOUBLE, name, &val, NULL, &status),
            "update_key"
        );
        return status;
    }
    int put_str(const char* name, const char* val) {
        int status = 0;
        check(
            fits_update_key(f, TSTRING, name, (void*)val, NULL, &status),
            "update_key"
        );
        return status;
    }

    // advance to next HDU
    //
    int next() {
        int status = 0;
        fits_movrel_hdu(f, 1, NULL, &status);
        return status;
    }

    // stuff related to images
    //
    int create_image(int type, int dim) {
        int status = 0;
        long dims[2] = {dim, dim};
        check(
            fits_create_img(f, fits_image_type[type], 2, dims, &status),
            "create_img"
        );
        return status;
    }

    int write_image(int type, int dim, void* data) {
        int status = 0;
        long nelements = dim*dim;
#if 1
        char *q = (char*)data;
        char *x[dim];
        for (int i=0; i<dim; i++) {
            x[i] = q+i*dim*nbytes[type];
        }
        long fpixel = 1;
        check(
            fits_write_img(f, fits_data_type[type], fpixel, dim*dim, x[0], &status),
            "write_pix"
        );
#else
        // the following should work but doesn't

        long fpixel[2] = {1,1};
        check(
            fits_write_pix(f, fits_data_type[type], fpixel, nelements, x, &status),
            "write_pix"
        );
#endif
        return status;
    }

    int read_image(int type, int dim, void* data) {
        int status = 0;
#if 1
        long fpixel = 1;
        long nbuffer = dim*dim;
        int nullval = 0;
        int anynull;
        check(
            fits_read_img(
                f, fits_data_type[type], fpixel, nbuffer, &nullval, data, &anynull, &status
            ),
            "read_img"
        );
#else
        // The following should work (?) but doesn't
        //
        long dims[2];
        int naxis, bitpix;
        long fpixel[2] = {1,1};

        check(
            fits_get_img_param(f, 2, &bitpix, &naxis, dims, &status),
            "get_img_params"
        );
        if (status) return status;
        if (bitpix != nbits[type]) {
            fprintf(stderr, "read_img(): unexpected nbits %d\n", bitpix);
            exit(1);
        }
        if (naxis != 2) {
            fprintf(stderr, "read_img(): unexpected ndims %d\n", naxis);
            exit(1);
        }
        if (dims[0] != dim || dims[1] != dim) {
            fprintf(stderr, "unexpected dims %ld %ld\n", dims[0], dims[1]);
            return -1;
        }
        check(
            fits_read_pix(f, fits_data_type[type], fpixel, nbits[type], NULL, data, NULL, &status),
            "read_pix"
        );
#endif
        return status;
    }
};

#endif

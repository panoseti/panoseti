// functions for reading/writing PanoSETI file format (.pff) files,
// and stuff related to file names
//
// A PFF file is a sequence of blocks,
// each of which is either text (e.g. a JSON doc) or a binary image.
// Each block starts with a "block type" byte (see values below).
// Text blocks are terminated with a blank line (two newlines in a row)
//
// A PFF file is not self-describing; you have to know what to expect,
// e.g. that images are 32x32.
// The type codes are mostly for sanity checking.
//
// See https://github.com/panoseti/panoseti/wiki/Data-file-format

#ifndef PANOSETI_PFF_H
#define PANOSETI_PFF_H

#include <string>
using std::string;

// block types
#define PFF_JSON_START '{'
#define PFF_IMAGE_START '*'

#define PFF_ERROR_BAD_TYPE  -1
#define PFF_ERROR_READ      -2

extern void pff_start_json(FILE* f);
extern void pff_end_json(FILE* f);
extern void pff_write_image(FILE* f, int nbytes, void* image);
extern int pff_read_json(FILE* f, string &s);
extern int pff_read_image(FILE* f, int nbytes, void* img);

extern bool ends_with(const char* s, const char* suffix);

////////// DIR AND FILE NAMES ////////////////

typedef enum {
    DP_BIT16_IMG = 1,       // this must be first
    DP_BIT8_IMG,
    DP_PH_256_IMG,
    DP_PH_1024_IMG,
    DP_NONE                 // this must be last
} DATA_PRODUCT;

inline int bytes_per_pixel(DATA_PRODUCT dp) {
    if (dp == DP_BIT16_IMG) return 2;
    if (dp == DP_BIT8_IMG) return 1;
    if (dp == DP_PH_256_IMG) return 2;
    if (dp == DP_PH_1024_IMG) return 2;
    return DP_BIT16_IMG;
}

// the info encoded in a dir name
//
struct DIRNAME_INFO {
    double start_time;      // UNIX time
    string observatory;
    string run_type;        // "SCI", "CAL" or "ENG"

    DIRNAME_INFO(){}
    DIRNAME_INFO(
        double _start_time, const string &_observatory, const string &_run_type
    ) {
        start_time = _start_time;
        observatory = _observatory;
        run_type = _run_type;
    }
    void make_dirname(string&);
    int parse_dirname(char*);
    void copy_to(DIRNAME_INFO* dip);
};

// the info encoded in a file name
//
struct FILENAME_INFO {
    double start_time;
    DATA_PRODUCT data_product;
    int bytes_per_pixel;
    int module;
    int seqno;

    FILENAME_INFO(){}
    FILENAME_INFO(
        double _start_time, DATA_PRODUCT _data_product, int _bytes_per_pixel,
        int _module, int _seqno
    ) {
        start_time = _start_time;
        data_product = _data_product;
        bytes_per_pixel = _bytes_per_pixel;
        module = _module;
        seqno = _seqno;
    }
    void make_filename(string&);
    int parse_filename(char*);
    int copy_to(FILENAME_INFO* fileInfo);
};

// given a string of the form .../d/f, return d and f
//
extern int pff_parse_path(const char* path, string& dir, string& file);
extern bool is_pff_file(const char*);

#endif

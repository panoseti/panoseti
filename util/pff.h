// functions for reading/writing PanoSETI file format (.pff) files,
// and stuff related to file names
//
// A PFF file is a sequence of blocks,
// each of which is either text (e.g. a JSON doc) or a binary image.
// Each block starts with a "block type" byte (see values below).
// Text blocks are terminated with a zero byte.
//
// A PFF file is not self-describing; you have to know what to expect,
// e.g. that images are 32x32.
// The type codes are mostly for sanity checking.

#ifndef PANOSETI_PFF_H
#define PANOSETI_PFF_H

#include <string>
using std::string;

#define PFF_TYPE_TEXT       1
#define PFF_TYPE_IMAGE      2

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
    DP_STATIC_META,
    DP_DYNAMIC_META,
    DP_BIT16_IMG,
    DP_BIT8_IMG,
    DP_PH_IMG
} DATA_PRODUCT;


// the info encoded in a dir name
//
struct DIRNAME_INFO {
    double start_time;
    char observatory[256];

    DIRNAME_INFO(){}
    DIRNAME_INFO(double _start_time, const char* _observatory) {
        start_time = _start_time;
        strcpy(observatory, _observatory);
    }
    void make_dirname(string&);
    int parse_dirname(char*);
};

// the info encoded in a file name
//
struct FILENAME_INFO {
    double start_time;
    DATA_PRODUCT data_product;
    int bytes_per_pixel;
    int dome;
    int module;
    int seqno;
    char *fileName;

    FILENAME_INFO(){}
    FILENAME_INFO(
        double _start_time, DATA_PRODUCT _data_product, int _bytes_per_pixel,
        int _dome, int _module, int _seqno
    ) {
        start_time = _start_time;
        data_product = _data_product;
        bytes_per_pixel = _bytes_per_pixel;
        dome = _dome;
        module = _module;
        seqno = _seqno;
    }
    void make_filename(string&);
    int parse_filename(char*);
};

// given a string of the form .../d/f, return d and f
//
extern int pff_parse_path(const char* path, string& dir, string& file);

#if 0
// the info for managing file pointers for all data products
//
struct FILE_PTRS{
    FILE *dynamicMeta, *bit16Img, *bit8Img, *PHImg;
    FILE_PTRS(const char *diskDir, DIRNAME_INFO *dirInfo, FILENAME_INFO *fileInfo, const char *mode);
};

////////// Structures for Reading and Parsing file in PFF////////////////

struct PF {
    DATA_PRODUCT dataProduct;
    FILE *filePtr;
    PF(FILENAME_INFO *fileInfo, DIRNAME_INFO *dirInfo);
    PF(const char *dirName, const char *fileName);
};
#endif

#endif

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

#include <string>
using std::string;

#define PFF_TYPE_TEXT       1
#define PFF_TYPE_IMAGE      2

#define PFF_ERROR_BAD_TYPE  -1
#define PFF_ERROR_READ      -2

inline void pff_start_json(FILE* f) {
    static char buf = PFF_TYPE_TEXT;
    fwrite(&buf, 1, 1, f);
}

inline void pff_end_json(FILE* f) {
    static char buf = 0;
    fwrite(&buf, 1, 1, f);
}

inline void pff_write_image(
    FILE* f, int nbytes, void* image
) {
    static char buf = PFF_TYPE_IMAGE;
    fwrite(&buf, 1, 1, f);
    fwrite(image, 1, nbytes, f);
}

int pff_read_json(FILE* f, string &s) {
    char c;
    if (fread(&c, 1, 1, f) != 1) {
        return PFF_ERROR_READ;
    }
    if (c != PFF_TYPE_TEXT) {
        return PFF_ERROR_BAD_TYPE;
    }
    s.clear();
    while(1) {
        c = fgetc(f);
        if (c == EOF) {
            return PFF_ERROR_READ;
        }
        if (c == 0) {
            break;
        }
        s.append(&c, 1);
    }
    return 0;
}

int pff_read_image(FILE* f, int nbytes, void* img) {
    char c;
    if (fread(&c, 1, 1, f) != 1) {
        return PFF_ERROR_READ;
    }
    if (c != PFF_TYPE_IMAGE) {
        return PFF_ERROR_BAD_TYPE;
    }
    if (fread(img, 1, nbytes, f) != nbytes) {
        return PFF_ERROR_READ;
    }
    return 0;
}

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

    DIRNAME_INFO();
    DIRNAME_INFO(double _start_time, const char* _observatory);
    void make(string&);
    int parse(char*);
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

    FILENAME_INFO();
    FILENAME_INFO(double _start_time, DATA_PRODUCT _data_product, int _bytes_per_pixel, int _dome, int _module, int _seqno);
    void make(string&);
    int parse(char*);
};

struct FILE_PTRS{
    FILE *dynamicMeta, *bit16Img, *bit8Img, *PHImg;
    FILE_PTRS(const char *diskDir, DIRNAME_INFO *dirInfo, FILENAME_INFO *fileInfo, const char *mode);
};
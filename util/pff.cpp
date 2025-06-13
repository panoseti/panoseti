// functions for reading/writing PanoSETI file format (.pff) files
// See pff.h

#include <string.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <time.h>

#include "pff.h"

using std::string;
using std::vector;

const char* dp_to_str(DATA_PRODUCT dp) {
    switch (dp) {
    case DP_BIT16_IMG: return "img16";
    case DP_BIT8_IMG: return "img8";
    case DP_PH_256_IMG: return "ph256";
    case DP_PH_1024_IMG: return "ph1024";
    }
    return "unknown";
}

DATA_PRODUCT str_to_dp(const char* s) {
    if (!strcmp(s, "img16")) return DP_BIT16_IMG;
    if (!strcmp(s, "img8")) return DP_BIT8_IMG;
    if (!strcmp(s, "ph256")) return DP_PH_256_IMG;
    if (!strcmp(s, "ph1024")) return DP_PH_1024_IMG;
    return DP_NONE;
}

void pff_start_json(FILE* f) {
}

void pff_end_json(FILE* f) {
    fprintf(f, "\n\n");
}

void pff_write_image(
    FILE* f, int nbytes, void* image
) {
    static char buf = PFF_IMAGE_START;
    fwrite(&buf, 1, 1, f);
    fwrite(image, 1, nbytes, f);
}

int pff_read_json(FILE* f, string &s) {
    char c;
    s.clear();
    while (1) {
        if (fread(&c, 1, 1, f) != 1) {
            return PFF_ERROR_READ;
        }
        if (c == '\n') continue;
        if (c == PFF_JSON_START) {
            break;
        }
        return PFF_ERROR_BAD_TYPE;
    }
    s.append(&c, 1);
    bool last_nl = false;   // last char was newline
    while(1) {
        c = fgetc(f);
        if (c == EOF) {
            return PFF_ERROR_READ;
        }
        if (c == '\n') {
            if (last_nl) {
                break;
            }
            last_nl = true;
        } else {
            last_nl= false;
        }
        s.append(&c, 1);
    }
    return 0;
}

int pff_read_image(FILE* f, int nbytes, void* img) {
    char c;
    while (1) {
        if (fread(&c, 1, 1, f) != 1) {
            return PFF_ERROR_READ;
        }
        if (c == PFF_IMAGE_START) {
            break;
        }
        return PFF_ERROR_BAD_TYPE;
    }
    if (fread(img, 1, nbytes, f) != nbytes) {
        return PFF_ERROR_READ;
    }
    return 0;
}

////////// DIR/FILE NAME STUFF ////////////////

// separator between name and value
//
#define VAL_SEP '_'

// separator between pairs
//
#define PAIR_SEP '.'

struct NV_PAIR {
    char name[64], value[256];
    int parse(const char *s) {
        char *p = (char*)strchr(s, VAL_SEP);
        if (!p) return -1;
        *p = 0;
        strcpy(name, s);
        strcpy(value, p+1);
        return 0;
    }
};

// get substrings separated by PAIR_SEP
//
void split_pair_sep(char *name, vector<string> &pieces) {
    char *p = name;
    while (1) {
        char *q = strchr(p, PAIR_SEP);
        if (!q) break;
        *q = 0;
        pieces.push_back(string(p));
        p = q+1;
    }
    pieces.push_back(string(p));
}

int pff_parse_path(const char* path, string& dir, string& file) {
    char buf[4096];
    strcpy(buf, path);
    char *p = strrchr(buf, '/');
    if (!p) return -1;
    file = p+1;
    *p = 0;
    p = strrchr(buf, '/');
    if (!p) return -1;
    dir = p+1;
    return 0;
}

bool ends_with(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    if (n<m) return false;
    return (strcmp(s+n-m, suffix)) == 0;
}

bool is_pff_file(const char* path) {
    return ends_with(path, ".pff");
}

void DIRNAME_INFO::make_dirname(string &s) {
    char buf[1024], tbuf[256];

    time_t x = (time_t)start_time;
    struct tm* tm = gmtime(&x);
    strftime(tbuf, sizeof(tbuf), "%FT%TZ", tm);
    sprintf(buf, "obs%c%s%cstart%c%s%cruntype%c%s",
        VAL_SEP, observatory.c_str(),
    PAIR_SEP, VAL_SEP, tbuf,
    PAIR_SEP, VAL_SEP, run_type.c_str()
    );
    s = buf;
}

int DIRNAME_INFO::parse_dirname(char* name) {
    vector<string> pieces;
    split_pair_sep(name, pieces);
    for (int i=0; i<pieces.size(); i++) {
        NV_PAIR nvp;
        int retval = nvp.parse(pieces[i].c_str());
        if (retval) {
            fprintf(stderr, "bad filename component: %s\n", pieces[i].c_str());
        }
        if (!strcmp(nvp.name, "obs")) {
            observatory = nvp.value;
        } else if (!strcmp(nvp.name, "runtype")) {
            run_type = nvp.value;
        } else if (!strcmp(nvp.name, "start")) {
            struct tm tm;
            char *p = strptime(nvp.value, "%FT%T%z", &tm);
            time_t t = mktime(&tm);
            start_time = (double)t;
        } else {
            fprintf(stderr, "unknown dirname key: %s\n", nvp.name);
        }
    }
    return 0;
}

void DIRNAME_INFO::copy_to(DIRNAME_INFO* dip){
    dip->start_time = start_time;
    dip->observatory = observatory;
    dip->run_type = run_type;
}

void FILENAME_INFO::make_filename(string &s) {
    char buf[1024], tbuf[256];

    time_t x = (time_t)start_time;
    struct tm* tm = gmtime(&x);
    strftime(tbuf, sizeof(tbuf), "%FT%TZ", tm);
    sprintf(buf, "start%c%s%cdp%c%s%cbpp%c%d%cmodule%c%d%cseqno%c%d.pff",
        VAL_SEP, tbuf,
        PAIR_SEP, VAL_SEP, dp_to_str(data_product),
        PAIR_SEP, VAL_SEP, bytes_per_pixel,
        PAIR_SEP, VAL_SEP, module,
        PAIR_SEP, VAL_SEP, seqno
    );
    s = buf;
}

int FILENAME_INFO::parse_filename(char* name) {
    vector<string> pieces;
    char* p = strrchr(name, '.');   // trim .pff
    if (!p) return 1;
    *p = 0;
    split_pair_sep(name, pieces);
    for (int i=0; i<pieces.size(); i++) {
        NV_PAIR nvp;
        int retval = nvp.parse(pieces[i].c_str());
        if (retval) {
            fprintf(stderr, "bad filename component: %s\n", pieces[i].c_str());
        }
        if (!strcmp(nvp.name, "start")) {
            struct tm tm;
            char *p = strptime(nvp.value, "%FT%T%z", &tm);
            time_t t = mktime(&tm);
            start_time = (double)t;
        } else if (!strcmp(nvp.name, "dp")) {
            data_product = str_to_dp(nvp.value);
        } else if (!strcmp(nvp.name, "bpp")) {
            bytes_per_pixel = atoi(nvp.value);
        } else if (!strcmp(nvp.name, "module")) {
            module = atoi(nvp.value);
        } else if (!strcmp(nvp.name, "seqno")) {
            seqno = atoi(nvp.value);
        } else {
            fprintf(stderr, "unknown filename key: %s\n", nvp.name);
        }
    }
    return 0;
}

int FILENAME_INFO::copy_to(FILENAME_INFO* fileInfo){
    fileInfo->start_time = this->start_time;
    fileInfo->data_product = this->data_product;
    fileInfo->bytes_per_pixel = this->bytes_per_pixel;
    fileInfo->module = this->module;
    fileInfo->seqno = this->seqno;
    return 1;
}

#if 0
int main(int, char**) {
    DIRNAME_INFO di(time(0), "Palomar", "SCI");
    di.observatory = "Palomar";
    di.run_type = "SCI";
    di.start_time = time(0);
    string s;
    di.make_dirname(s);

    char buf[256];
    strcpy(buf, s.c_str());
    printf("dir name: %s\n", buf);

    di.parse_dirname(buf);
    printf("parsed: obs %s type %s time %f\n", di.observatory.c_str(),
    di.run_type.c_str(), di.start_time);

    FILENAME_INFO fi;
    fi.start_time = time(0);
    fi.data_product = DP_PH_IMG;
    fi.bytes_per_pixel = 2;
    fi.module=14;
    fi.seqno = 5;
    fi.make_filename(s);

    strcpy(buf, s.c_str());
    printf("file name: %s\n", buf);

    fi.parse_filename(buf);
    printf("parsed: time %f dp %d bpp %d module %d seqno %d\n",
        fi.start_time, fi.data_product, fi.bytes_per_pixel,
        fi.module, fi.seqno
    );
}
#endif

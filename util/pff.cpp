#include <string.h>
#include <string>
#include <vector>
#include <stdio.h>

#include "pff.h"

using std::string;
using std::vector;

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

struct NV_PAIR {
    char name[64], value[256];
    int parse(const char *s) {
        char *p = (char*)strchr(s, '=');
        if (!p) return -1;
        *p = 0;
        strcpy(name, s);
        strcpy(value, p+1);
        return 0;
    }
};

// get comma-separated substrings
//
void split_comma(char *name, vector<string> &pieces) {
    char *p = name;
    while (1) {
        char *q = strchr(p, ',');
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

void DIRNAME_INFO::make_dirname(string &s) {
    char buf[1024], tbuf[256];

    time_t x = (time_t)start_time;
    struct tm* tm = localtime(&x);
    strftime(tbuf, sizeof(tbuf), "%a_%b_%d_%T_%Y", tm);
    sprintf(buf, "obs=%s,start=%s,run_type=%s",
        observatory.c_str(), tbuf, run_type.c_str()
    );
    s = buf;
}

int DIRNAME_INFO::parse_dirname(char* name) {
    vector<string> pieces;
    split_comma(name, pieces);
    for (int i=0; i<pieces.size(); i++) {
        NV_PAIR nvp;
        int retval = nvp.parse(pieces[i].c_str());
        if (retval) {
            fprintf(stderr, "bad filename component: %s\n", pieces[i].c_str());
        }
        if (!strcmp(nvp.name, "obs")) {
            observatory = nvp.value;
        } else if (!strcmp(nvp.name, "run_type")) {
            run_type = nvp.value;
        } else if (!strcmp(nvp.name, "st")) {
            struct tm tm;
            char *p = strptime(nvp.value, "%a_%b_%d_%T_%Y", &tm);
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
    struct tm* tm = localtime(&x);
    strftime(tbuf, sizeof(tbuf), "%a_%b_%d_%T_%Y", tm);
    sprintf(buf, "start=%s,dp=%d,bpp=%d,dome=%d,module=%d,seqno=%d.pff",
        tbuf, data_product, bytes_per_pixel, dome, module, seqno
    );
    s = buf;
}

int FILENAME_INFO::parse_filename(char* name) {
    vector<string> pieces;
    char* p = strrchr(name, '.');   // trim .pff
    if (!p) return 1;
    *p = 0;
    split_comma(name, pieces);
    for (int i=0; i<pieces.size(); i++) {
        NV_PAIR nvp;
        int retval = nvp.parse(pieces[i].c_str());
        if (retval) {
            fprintf(stderr, "bad filename component: %s\n", pieces[i].c_str());
        }
        if (!strcmp(nvp.name, "st")) {
            struct tm tm;
            char *p = strptime(nvp.value, "%a_%b_%d_%T_%Y", &tm);
            time_t t = mktime(&tm);
            start_time = (double)t;
        } else if (!strcmp(nvp.name, "dp")) {
            data_product = (DATA_PRODUCT)atoi(nvp.value);
        } else if (!strcmp(nvp.name, "dome")) {
            dome = atoi(nvp.value);
        } else if (!strcmp(nvp.name, "mod")) {
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
    fileInfo->dome = this->dome;
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
    printf("dir name: %s\n", s.c_str());

    FILENAME_INFO fi;
    fi.start_time = time(0);
    fi.data_product = DP_PH_IMG;
    fi.bytes_per_pixel = 2;
    fi.dome = 0;
    fi.module=14;
    fi.seqno = 5;
    fi.make_filename(s);
    printf("file name: %s\n", s.c_str());

    char buf[256];
    strcpy(buf, "obs=Palomar,start=Fri_Aug_27_15:21:46_2021");
    di.parse_dirname(buf);

    strcpy(buf, "start=Fri_Aug_27_15:21:46_2021,dp=1,bpp=2,dome=0,module=14,seqno=5.pff");
    fi.parse_filename(buf);
}
#endif

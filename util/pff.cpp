#include <string.h>
#include <string>
#include <vector>
#include <stdio.h>

#include "pff.h"

using std::string;
using std::vector;

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

DIRNAME_INFO::DIRNAME_INFO(){}
DIRNAME_INFO::DIRNAME_INFO(double _start_time, const char* _observatory){
    start_time = _start_time;
    strcpy(observatory, _observatory);
}

void DIRNAME_INFO::make(string &s) {
    char buf[1024], tbuf[256];

    time_t x = (time_t)start_time;
    struct tm* tm = localtime(&x);
    strftime(tbuf, sizeof(tbuf), "%a_%b_%d_%T_%Y", tm);
    sprintf(buf, "obs=%s,st=%s", observatory, tbuf);
    s = buf;
}

int DIRNAME_INFO::parse(char* name) {
    vector<string> pieces;
    split_comma(name, pieces);
    for (int i=0; i<pieces.size(); i++) {
        NV_PAIR nvp;
        int retval = nvp.parse(pieces[i].c_str());
        if (retval) {
            fprintf(stderr, "bad filename component: %s\n", pieces[i].c_str());
        }
        if (!strcmp(nvp.name, "obs")) {
            strcpy(observatory, nvp.value);
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

FILENAME_INFO::FILENAME_INFO(){}
FILENAME_INFO::FILENAME_INFO(double _start_time, DATA_PRODUCT _data_product, int _bytes_per_pixel, int _dome, int _module, int _seqno) {
    FILENAME_INFO value;
    value.start_time = _start_time;
    value.data_product = _data_product;
    value.bytes_per_pixel = _bytes_per_pixel;
    value.dome = _dome;
    value.module = _module;
    value.seqno = _seqno;
}

void FILENAME_INFO::make(string &s) {
    char buf[1024], tbuf[256];

    time_t x = (time_t)start_time;
    struct tm* tm = localtime(&x);
    strftime(tbuf, sizeof(tbuf), "%a_%b_%d_%T_%Y", tm);
    sprintf(buf, "st=%s,dp=%d,bpp=%d,dome=%d,module=%d,seqno=%d.pff",
        tbuf, data_product, bytes_per_pixel, dome, module, seqno
    );
    s = buf;
}

int FILENAME_INFO::parse(char* name) {
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

FILE_PTRS::FILE_PTRS(const char *diskDir, DIRNAME_INFO dirInfo, FILENAME_INFO fileInfo, const char *mode){
    string fileName;
    string dirName;
    dirInfo.make(dirName);
    dirName = diskDir + dirName + "/";
    mkdir(dirName.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    

    for (int dp = DP_DYNAMIC_META; dp <= DP_PH_IMG; dp++){
        fileInfo.data_product = (DATA_PRODUCT)dp;
        fileInfo.make(fileName);
        switch (dp){
            case DP_DYNAMIC_META:
                dynamicMeta = fopen((dirName + fileName).c_str(), mode);
                break;
            case DP_BIT16_IMG:
                bit16Img = fopen((dirName + fileName).c_str(), mode);
                break;
            case DP_BIT8_IMG:
                bit8Img = fopen((dirName + fileName).c_str(), mode);
                break;
            case DP_PH_IMG:
                PHImg = fopen((dirName + fileName).c_str(), mode);
                break;
            default:
                break;
        }
        if (access(dirName.c_str(), F_OK) == -1) {
            printf("Error: Unable to access file - %s\n", dirName.c_str());
            exit(0);
        }
        printf("Created file %s\n", (dirName + fileName).c_str());
    }
}

#if 1
int main(int, char**) {
    DIRNAME_INFO di;
    strcpy(di.observatory, "Palomar");
    di.start_time = time(0);
    string s;
    di.make(s);
    printf("dir name: %s\n", s.c_str());

    FILENAME_INFO fi;
    fi.start_time = time(0);
    fi.data_product = DP_PH_IMG;
    fi.bytes_per_pixel = 2;
    fi.dome = 0;
    fi.module=14;
    fi.seqno = 5;
    fi.make(s);
    printf("file name: %s\n", s.c_str());

    char buf[256];
    strcpy(buf, "obs=Palomar,st=Fri_Aug_27_15:21:46_2021");
    di.parse(buf);

    strcpy(buf, "st=Fri_Aug_27_15:21:46_2021,dp=1,bpp=2,dome=0,module=14,seqno=5.pff");
    fi.parse(buf);
}
#endif

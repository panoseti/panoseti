#include "rapidjson/include/rapidjson/document.h"
#include "rapidjson/include/rapidjson/stringbuffer.h"

using namespace rapidjson;

#include "pff.h"

struct FOO {
    double x;
    char blah[256];

    void write_json(FILE *f) {
        fprintf(f, "{\"x\": %f,\n\"s\": \"%s\"}", x, blah);
    }
    int read_json(string &s) {
        Document d;
        if (d.Parse(s.c_str()).HasParseError()) {
            return -1;
        }
        x = d["x"].GetDouble();
        strcpy(blah, d["s"].GetString());
        return 0;
    }
};

int main(int, char**) {
    short img[1024];
    FOO foo;
    foo.x = 17;
    strcpy(foo.blah, "foobar");

    FILE* f = fopen("test.pff", "w");
    pff_start_json(f);
    foo.write_json(f);
    pff_end_json(f);
    pff_write_image(f, 1024*sizeof(short), img);
    fclose(f);
}

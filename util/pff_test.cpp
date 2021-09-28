// test program for PFF API:
// writes a file "test.pff", then reads it

#include "rapidjson/include/rapidjson/document.h"
#include "rapidjson/include/rapidjson/stringbuffer.h"

using namespace rapidjson;

#include "pff.h"

// example of converting a structure to/from JSON
//
struct FOO {
    double x;
    char blah[256];

    // could use Rapidjson for this but easier to do it ourselves
    //
    void write_json(FILE *f) {
        fprintf(f, "{\"x\": %f,\n\"s\": \"%s\"}", x, blah);
    }

    int parse_json(string &s) {
        Document d;
        if (d.Parse(s.c_str()).HasParseError()) {
            return -1;
        }
        // get elements; check for missing values
        //
        x = d.HasMember("x")?d["x"].GetDouble():-1;
        strcpy(blah, d.HasMember("s")?d["s"].GetString():"undefined");
        return 0;
    }
};

void write_file() {
    short img[1024];
    FOO foo;
    foo.x = 17;
    strcpy(foo.blah, "foobar");
    for (int i=0; i<1024; i++) {
        img[i] = i;
    }

    FILE *f = fopen("test.pff", "w");
    pff_start_json(f);
    foo.write_json(f);
    pff_end_json(f);
    pff_write_image(f, 1024*sizeof(short), img);
    fclose(f);
}

void read_file() {
    FOO foo;
    short img[1024];

    FILE *f = fopen("test.pff", "r");
    string s;
    int retval = pff_read_json(f, s);
    foo.parse_json(s);
    printf("header: %f %s\n", foo.x, foo.blah);
    retval = pff_read_image(f, 1024*sizeof(short), img);
    for (int i=0; i<32; i++) {
        for (int j=0; j<32; j++) {
            printf("%d ", img[i*32+j]);
        }
        printf("\n");
    }
    fclose(f);
}

int main(int, char**) {
    write_file();
    read_file();
}

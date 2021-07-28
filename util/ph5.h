// functions for getting data from a PanoSETI HDF5 file
//
// integer return values are error codes; nonzero = error

#include <string>

#include "hdf5.h"

using std::string;

// a "frame group" is ~5000 frames.
// a file can have arbitrarily many
//
struct FRAME_GROUP {
    uint16_t *data;
    int nframes;
    uint16_t* get_frame(int iframe, int module, int quabo) {
        return data + iframe*(8*64) + module*256 + quabo*64;
    }
    FRAME_GROUP(){
        data = 0;
    }
    ~FRAME_GROUP() {
        if (data) {
            //printf("freeing %lx\n", data);
            free(data);
        }
    }
};

// represents a PanoSETI HDF5 file
//
struct PH5 {
    hid_t file_id;

    int open(const char* path);
    void close();
    int get_attr(const char* name, string&);
    int get_frame_group( const char* base_name, int igroup, FRAME_GROUP&);
};

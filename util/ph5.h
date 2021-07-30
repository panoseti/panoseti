#ifndef PH5_H
#define PH5_H

// functions for getting data from a PanoSETI HDF5 file
//
// integer return values are error codes; nonzero = error

#include <string>

#include "hdf5.h"

using std::string;

// terminology:
// "qframe": a frame from a quabo.  16x16 pixels
// 'mframe": a frame from a mobo.  32x32 pixels
//      NOTE: because of hardware variation, you may have
//      to rotate some of the qframes within an mframe
// 'frame pair': a pair of corresponding mframes from 2 domes
// 'frame set': a sequence of frame pairs, typically 5000,
//      stored as an HDF5 dataset
// a file can have arbitrarily many frame sets
//

#define MFRAME_DIM  32
#define MFRAME_PIXELS   1024
#define FRAME_PAIR_PIXELS 2*MFRAME_PIXELS

struct FRAME_SET {
    uint16_t *data;
    int nframe_pairs;

    uint16_t* get_mframe(int iframe_pair, int module) {
        return data + iframe_pair*FRAME_PAIR_PIXELS + module*MFRAME_PIXELS;
    }
    FRAME_SET(){
        data = 0;
    }
    ~FRAME_SET() {
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
    int get_frame_set( const char* base_name, int iset, FRAME_SET&);
};

#endif

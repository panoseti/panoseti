#ifndef _COMPUTE_THREAD_H_
#define _COMPUTE_THREAD_H_

#include "databuf.h"

// structure used by the compute thread to accumulate a module image;
// only some of the quabo sub-images may be present
//
struct MODULE_IMAGE_BUFFER {
    uint32_t max_nanosec;       // min/max arrival times of quabo images
    uint32_t min_nanosec;
    char quabos_bitmap;         // bitmap for which quabos images are present
    MODULE_IMAGE_HEADER mod_head;   // packet headers stored here
    uint8_t data[BYTES_PER_MODULE_FRAME];
    MODULE_IMAGE_BUFFER() {
        clear();
    }
    void clear(){
        memset(this, 0, sizeof(*this));
    }
};

// structure used by the compute thread to accumulate a pulse-height image;
// only some of the quabo sub-images may be present
//
struct PH_IMAGE_BUFFER {
    uint32_t max_nanosec;       // min/max arrival times of quabo images
    uint32_t min_nanosec;
    char quabos_bitmap;         // bitmap for which quabos images are present
    PH_IMAGE_HEADER ph_head;   // packet headers stored here
    uint8_t data[BYTES_PER_PH_FRAME];
    PH_IMAGE_BUFFER() {
        clear();
    }
    void clear(){
        memset(this, 0, sizeof(*this));
    }
};

// structure used by the compute thread to accumulate multiple pulse-height images
// in a circular buffer of PH_IMAGE_BUFFERs.
// 
struct CIRCULAR_PH_IMAGE_BUFFER {
    PH_IMAGE_BUFFER* buf[CIRCULAR_PH_BUFFER_LENGTH];
    int first, last;
    bool partial_PH1024_image_write = false;
    CIRCULAR_PH_IMAGE_BUFFER() {
        first = last = 0;
        for (int i = 0; i < CIRCULAR_PH_BUFFER_LENGTH; i++) {
            buf[i] = new PH_IMAGE_BUFFER();
        }
    }
};
#endif

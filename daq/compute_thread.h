#ifndef _COMPUTE_THREAD_H_
#define _COMPUTE_THREAD_H_

#include "databuf.h"

// structure used by the compute thread to accumulate a module image;
// only some of the quabo sub-images may be present

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

#endif

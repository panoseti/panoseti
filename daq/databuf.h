#ifndef _DATABUF_H_
#define _DATABUF_H_

// Panoseti hashpipe data structures

#include <string>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "hashpipe.h"
#include "hashpipe_databuf.h"


// the sizes of packets images and headers in bytes

#define BYTES_PER_PKT_IMAGE         512
    // Number of bytes for a 16 bit  packet
#define BYTES_PER_8BIT_PKT_IMAGE    256
    // Number of bytes for a 8 bit image packet
#define BYTE_PKT_HEADER             16
    // Number of bytes for the header for all packets

// the size and characteristics of the input and output circular buffers

#define CACHE_ALIGNMENT             256
    // Align the cache within the buffer
#define N_INPUT_BLOCKS              256
    // Number of blocks in the input buffer
#define N_OUTPUT_BLOCKS             64
    // Number of blocks in the output buffer
#define IN_PKT_PER_BLOCK            16384
    // Number of input packets stored in each block of the input buffer
#define OUT_MOD_PER_BLOCK           16384
    // Max Number of Modules stored in each block of the output buffer
#define OUT_COINC_PER_BLOCK         16384
    // Max Number of coincidence packets stored in each block of the output buffer

// Imaging Data Values and characteristics of modules

#define QUABO_PER_MODULE        4
    // Max Number of Quabos associated with a Module
#define PIXELS_PER_IMAGE        256
    // Number of pixels for each image data
#define BYTES_PER_MODULE_FRAME  QUABO_PER_MODULE*PIXELS_PER_IMAGE*2
    // Size of module image allocated in buffer

// the Block Sizes for the Input and Ouput Buffers

#define BYTES_PER_INPUT_IMAGE_BLOCK     IN_PKT_PER_BLOCK*BYTES_PER_PKT_IMAGE
    // Byte size of input image block.
    // Contains images for packets excluding headers
#define BYTES_PER_OUTPUT_FRAME_BLOCK    OUT_MOD_PER_BLOCK*BYTES_PER_MODULE_FRAME
    // Byte size of output frame block.
    // Contains frames for modules excluding headers
#define BYTES_PER_OUTPUT_COINC_BLOCK    OUT_COINC_PER_BLOCK*BYTES_PER_PKT_IMAGE
    // Byte size of output coincidence block.
    // Contains frames for coincidence packets excluding headers

// the algorithm constants for the hashpipe framework threads.
// Nanosecond threshold is used for syncing and grouping packets
// that are being collected by the network thread.
// The nanosecond value in the header values are read and if a new packet
// for an existing module is received the
// difference between the largest and the smallest nanosecond values
// (nanosecond value of the new packet is included in the calculation)
// is calculated and must be smaller than the nanosecond threshold.
// If the new packet's nanosecond value causes the difference
// to exceed the threshold then the old module data is flushed
// and the new module data is created starting with the new packet.
// More information can be seen in the compute thread.

#define NANOSEC_THRESHOLD        100
    // Nanosecond threshold used for grouping quabo images

// Module index is used for defining the array for storing pointers
// of module structures for both compute and output threads.
#define MAX_MODULE_INDEX         0xffff
    // Largest Module Index for compute and output thread

// conguration default file name
#define CONFIGFILE_DEFAULT "./module.config"
    // Default Location used for module config file

// the string buffer size
#define STR_BUFFER_SIZE 256

// the values from a packet header
// 
struct PACKET_HEADER {
    char acq_mode;
    uint16_t pkt_num;
    uint16_t mod_num;       // 0..255
    uint8_t quabo_num;        // 0..3
    uint32_t pkt_utc;
    uint32_t pkt_nsec;
    long int tv_sec;
    long int tv_usec;
    std::string toString(){
        return "acq_mode = " + std::to_string(this->acq_mode) +
                " pkt_num = " + std::to_string(this->pkt_num) +
                " mod_num = " + std::to_string(this->mod_num) +
                " quabo_num = " + std::to_string(this->quabo_num) +
                " pkt_utc = " + std::to_string(this->pkt_utc) +
                " pkt_nsec = " + std::to_string(this->pkt_nsec) +
                " tv_sec = " + std::to_string(this->tv_sec) +
                " tv_sec = " + std::to_string(this->tv_usec);
    };
};

// info about a module image:
// - the packet headers for the 4 quabo images comprising it
// - the module number
// produced by the compute thread, consumed by the output thread
// 
struct MODULE_IMAGE_HEADER {
    int bits_per_pixel;
    uint16_t mod_num;
    PACKET_HEADER pkt_head[QUABO_PER_MODULE];
    std::string toString(){
        std::string return_string = "bits_per_pixel = " + std::to_string(this->bits_per_pixel) + "\n";
        return_string += "mod_num = " + std::to_string(this->mod_num);
        for (int i = 0; i < QUABO_PER_MODULE; i++){
            return_string += "\n" + pkt_head[i].toString();
        }
        return return_string;
    }
};


// INPUT BUFFER STRUCTURES

// Input block header containing header information for the input buffer.

typedef struct HSD_input_block_header {
    uint64_t mcnt;                              // mcount of first packet
    PACKET_HEADER pkt_head[IN_PKT_PER_BLOCK];
    int n_pkts_in_block;
    int INTSIG;
} HSD_input_block_header_t;

typedef uint8_t HSD_input_header_cache_alignment[
    CACHE_ALIGNMENT - (sizeof(HSD_input_block_header_t)%CACHE_ALIGNMENT)
];

// Input data block within the input buffer. Contains image data within
// data_block and their header information within header.

typedef struct HSD_input_block {
    HSD_input_block_header_t header;
    HSD_input_header_cache_alignment padding;
    char data_block[BYTES_PER_INPUT_IMAGE_BLOCK];
} HSD_input_block_t;

// Input data buffer containing mutiple data blocks to be passed over to 
// compute thread for processing.

typedef struct HSD_input_databuf {
    hashpipe_databuf_t header;
    HSD_input_header_cache_alignment padding;
    HSD_input_block_t block[N_INPUT_BLOCKS];
} HSD_input_databuf_t;

// OUTPUT BUFFER STRUCTURES

// Output block header containing header information for the data streams 
// created by the compute thread.

typedef struct HSD_output_block_header {
    uint64_t mcnt;
    MODULE_IMAGE_HEADER img_mod_head[OUT_MOD_PER_BLOCK];
    int n_img_module;

    PACKET_HEADER coinc_pkt_head[OUT_COINC_PER_BLOCK];
    int n_coinc_img;

    int INTSIG;
} HSD_output_block_header_t;

typedef uint8_t HSD_output_header_cache_alignment[
    CACHE_ALIGNMENT - (sizeof(HSD_output_block_header_t)%CACHE_ALIGNMENT)
];

// Output data block within the output buffer.
// Contains image and PH data generated by the compute thread.

typedef struct HSD_output_block {
    HSD_output_block_header_t header;
    HSD_output_header_cache_alignment padding;
    char img_block[BYTES_PER_OUTPUT_FRAME_BLOCK];
    char coinc_block[BYTES_PER_OUTPUT_COINC_BLOCK];
} HSD_output_block_t;

// Output data buffer containing multiple data blocks
// to be passed to output thread for disk writes.

typedef struct HSD_output_databuf {
    hashpipe_databuf_t header;
    HSD_output_header_cache_alignment padding;
    HSD_output_block_t block[N_OUTPUT_BLOCKS];
} HSD_output_databuf_t;

// INPUT BUFFER FUNCTIONS FROM HASHPIPE LIBRARY

hashpipe_databuf_t * HSD_input_databuf_create(int instance_id, int databuf_id);

// Input databuf attach
static inline HSD_input_databuf_t *HSD_input_databuf_attach(
    int instance_id, int databuf_id
){
    return (HSD_input_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

// Input databuf detach
static inline int HSD_input_databuf_detach(HSD_input_databuf_t *d){
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

// Input databuf clear
static inline void HSD_input_databuf_clear(HSD_input_databuf_t *d){
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

// Input databuf block status
static inline int HSD_input_databuf_block_status(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

// Input databuf total status
static inline int HSD_input_databuf_total_status(HSD_input_databuf_t *d){
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

// Input databuf wait free
static inline int HSD_input_databuf_wait_free(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

// Input databuf busy wait free
static inline int HSD_input_databuf_busywait_free(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

// Input databuf wait filled
static inline int HSD_input_databuf_wait_filled(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

// Input databuf busy wait filled
static inline int HSD_input_databuf_busywait_filled(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

// Input databuf set free
static inline int HSD_input_databuf_set_free(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

// Input databuf set filled
static inline int HSD_input_databuf_set_filled(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

// OUTPUT BUFFER FUNCTIONS FROM HASHPIPE LIBRARY

hashpipe_databuf_t *HSD_output_databuf_create(int instance_id, int databuf_id);

// Output databuf clear
static inline void HSD_output_databuf_clear(HSD_output_databuf_t *d){
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

// Output databuf attach
static inline HSD_output_databuf_t *HSD_output_databuf_attach(int instance_id, int databuf_id){
    return (HSD_output_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

// Output databuf detach
static inline int HSD_output_databuf_detach (HSD_output_databuf_t *d){
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

// Output block status
static inline int HSD_output_databuf_block_status(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

// Output databuf total status
static inline int HSD_output_databuf_total_status(HSD_output_databuf_t *d){
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

// Output databuf wait free
static inline int HSD_output_databuf_wait_free(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

// Output databuf busy wait free
static inline int HSD_output_databuf_busywait_free(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

// Output databuf wait filled
static inline int HSD_output_databuf_wait_filled(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

// Output databuf busy wait filled
static inline int HSD_output_databuf_busywait_filled(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

// Output databuf set free
static inline int HSD_output_databuf_set_free(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

// Output databuf set filled
static inline int HSD_output_databuf_set_filled(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

#endif

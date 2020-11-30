#include <stdint.h>
#include <stdio.h>
#include "hashpipe.h"
#include "hashpipe_databuf.h"
//Defining size of packets
#define PKTSIZE 528     //byte of packet size
#define BIT8PKTSIZE 272 //byte of 8bit packet size

//Defining the characteristics of the circuluar buffers
#define CACHE_ALIGNMENT         256
#define N_INPUT_BLOCKS          4                       //Number of blocks in the input buffer
#define N_OUTPUT_BLOCKS         8                       //Number of blocks in the output buffer
#define N_PKT_PER_BLOCK         40                      //Number of Pkt stored in each block
#define BLOCKSIZE               PKTSIZE*N_PKT_PER_BLOCK //Block size in bytes

/* INPUT BUFFER STRUCTURES */
typedef struct HSD_input_block_header {
    uint64_t mcnt;                              // mcount of first packet
    long int tv_sec[N_PKT_PER_BLOCK];
    long int tv_usec[N_PKT_PER_BLOCK];
} HSD_input_block_header_t;

typedef uint8_t HSD_input_header_cache_alignment[
    CACHE_ALIGNMENT - (sizeof(HSD_input_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct HSD_input_block {
    HSD_input_block_header_t header;
    HSD_input_header_cache_alignment padding;   // Maintain cache alignment
    char data_block[BLOCKSIZE*sizeof(char)];    //define input buffer
} HSD_input_block_t;

typedef struct HSD_input_databuf {
    hashpipe_databuf_t header;
    HSD_input_header_cache_alignment padding;   // Maintain chache alignment
    HSD_input_block_t block[N_INPUT_BLOCKS];
} HSD_input_databuf_t;

/*
 *  OUTPUT BUFFER STRUCTURES
 */
typedef struct HSD_output_block_header {
    uint64_t mcnt;
    long int tv_sec[N_PKT_PER_BLOCK];
    long int tv_usec[N_PKT_PER_BLOCK];
} HSD_output_block_header_t;

typedef uint8_t HSD_output_header_cache_alignment[
    CACHE_ALIGNMENT - (sizeof(HSD_output_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct HSD_output_block {
    HSD_output_block_header_t header;
    HSD_output_header_cache_alignment padding;  //Maintain cache alignment
    char result_block[BLOCKSIZE*sizeof(char)];
} HSD_output_block_t;

typedef struct HSD_output_databuf {
    hashpipe_databuf_t header;
    HSD_output_header_cache_alignment padding;
    HSD_output_block_t block[N_OUTPUT_BLOCKS];
} HSD_output_databuf_t;

/*
 * INPUT BUFFER FUNCTIONS FROM HASHPIPE LIBRARY
 */
hashpipe_databuf_t * HSD_input_databuf_create(int instance_id, int databuf_id);

//Input databuf attach
static inline HSD_input_databuf_t *HSD_input_databuf_attach(int instance_id, int databuf_id){
    return (HSD_input_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

//Input databuf detach
static inline int HSD_input_databuf_detach(HSD_input_databuf_t *d){
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

//Input databuf clear
static inline void HSD_input_databuf_clear(HSD_input_databuf_t *d){
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

//Input databuf block status
static inline int HSD_input_databuf_block_status(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

//Input databuf total status
static inline int HSD_input_databuf_total_status(HSD_input_databuf_t *d){
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

//Input databuf wait free
static inline int HSD_input_databuf_wait_free(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

//Input databuf busy wait free
static inline int HSD_input_databuf_busywait_free(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

//Input databuf wait filled
static inline int HSD_input_databuf_wait_filled(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

//Input databuf busy wait filled
static inline int HSD_input_databuf_busywait_filled(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

//Input databuf set free
static inline int HSD_input_databuf_set_free(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

//Input databuf set filled
static inline int HSD_input_databuf_set_filled(HSD_input_databuf_t *d, int block_id){
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

/*
 * OUTPUT BUFFER FUNCTIONS FROM HASHPIPE LIBRARY
 */

hashpipe_databuf_t *HSD_output_databuf_create(int instance_id, int databuf_id);

//Output databuf clear
static inline void HSD_output_databuf_clear(HSD_output_databuf_t *d){
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

//Output databuf attach
static inline HSD_output_databuf_t *HSD_output_databuf_attach(int instance_id, int databuf_id){
    return (HSD_output_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

//Output databuf detach
static inline int HSD_output_databuf_detach (HSD_output_databuf_t *d){
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

//Output block status
static inline int HSD_output_databuf_block_status(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

//Output databuf total status
static inline int HSD_output_databuf_total_status(HSD_output_databuf_t *d){
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

//Output databuf wait free
static inline int HSD_output_databuf_wait_free(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

//Output databuf busy wait free
static inline int HSD_output_databuf_busywait_free(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

//Output databuf wait filled
static inline int HSD_output_databuf_wait_filled(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

//Output databuf busy wait filled
static inline int HSD_output_databuf_busywait_filled(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

//Output databuf set free
static inline int HSD_output_databuf_set_free(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

//Output databuf set filled
static inline int HSD_output_databuf_set_filled(HSD_output_databuf_t *d, int block_id){
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}
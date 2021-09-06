#include <stdint.h>
#include <stdio.h>
#include "hashpipe.h"
#include "hashpipe_databuf.h"
#include "hdf5.h"
#include "hdf5_hl.h"


//Defining size of packets
#define PKTDATASIZE         512     //byte of data block
#define BIT8PKTDATASIZE     256     //byte of 8bit data block
#define HEADERSIZE          16      //byte of header

//Defining the characteristics of the circuluar buffers
#define CACHE_ALIGNMENT         256
#define N_INPUT_BLOCKS          4                       //Number of blocks in the input buffer
#define N_OUTPUT_BLOCKS         8                       //Number of blocks in the output buffer
#define IN_PKT_PER_BLOCK        320                      //Number of Pkt stored in each block
#define OUT_MODPAIR_PER_BLOCK   320                      //Max Number of Module Pairs stored in each block
#define COINC_PKT_PER_BLOCK     320                      //Max Number of Coinc packets stored in each block

//Defining Imaging Data Values
#define QUABOPERMODULE          4
#define PKTPERPAIR              QUABOPERMODULE*2
#define SCIDATASIZE             256
#define MODPAIRDATASIZE         PKTPERPAIR*SCIDATASIZE*2
#define PKTPERDATASET           5000                     //Number Module Pair data per dataset in HDF5 file

//Defining the Block Sizes for the Input and Ouput Buffers
#define INPUTBLOCKSIZE          IN_PKT_PER_BLOCK*PKTDATASIZE                    //Input Block size includes headers
#define OUTPUTBLOCKSIZE         OUT_MODPAIR_PER_BLOCK*MODPAIRDATASIZE           //Output Stream Block size excludes headers
#define OUTPUTCOICBLOCKSIZE     COINC_PKT_PER_BLOCK*PKTDATASIZE                 //Output Coinc Block size excluding headers


#define BLOCKSIZE           INPUTBLOCKSIZE


//Definng the numerical values
#define RANK                    3
#define HKDATASIZE              464
#define DATABLOCKSIZE           SCIDATASIZE*PKTPERPAIR+64+16
#define HKFIELDS                27
#define GPSFIELDS               10
#define NANOSECTHRESHOLD        20
#define MODULEINDEXSIZE         0xffff

#define MODULEPAIR_FORMAT "ModulePair_%05u_%05u"
#define DATAFILE_DEFAULT "./"
#define CONFIGFILE_DEFAULT "./module.config"

//TODO needs to be removed after changing compute thread
#define CONFIGFILE "./modulePair.config"

//Defining the string buffer size
#define STRBUFFSIZE 256




/* INPUT BUFFER STRUCTURES */
typedef struct HSD_input_block_header {
    uint64_t mcnt;                              // mcount of first packet
    char acqmode[IN_PKT_PER_BLOCK];
    uint16_t pktNum[IN_PKT_PER_BLOCK];
    uint16_t modNum[IN_PKT_PER_BLOCK];
    uint8_t quaNum[IN_PKT_PER_BLOCK];
    uint32_t pktUTC[IN_PKT_PER_BLOCK];
    uint32_t pktNSEC[IN_PKT_PER_BLOCK];
    long int tv_sec[IN_PKT_PER_BLOCK];
    long int tv_usec[IN_PKT_PER_BLOCK];
    int data_block_size;
    int INTSIG;
} HSD_input_block_header_t;

typedef uint8_t HSD_input_header_cache_alignment[
    CACHE_ALIGNMENT - (sizeof(HSD_input_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct HSD_input_block {
    HSD_input_block_header_t header;
    HSD_input_header_cache_alignment padding;       // Maintain cache alignment
    char data_block[INPUTBLOCKSIZE*sizeof(char)];   //define input buffer
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

    uint16_t modNum[OUT_MODPAIR_PER_BLOCK*2];
    char acqmode[OUT_MODPAIR_PER_BLOCK];
    //uint32_t pktUTC[OUT_MODPAIR_PER_BLOCK*PKTPERPAIR];
    uint16_t pktNum[OUT_MODPAIR_PER_BLOCK*PKTPERPAIR];
    uint32_t pktNSEC[OUT_MODPAIR_PER_BLOCK*PKTPERPAIR];
    long int tv_sec[OUT_MODPAIR_PER_BLOCK*PKTPERPAIR];
    long int tv_usec[OUT_MODPAIR_PER_BLOCK*PKTPERPAIR];
    uint8_t status[OUT_MODPAIR_PER_BLOCK];
    int stream_block_size;

    
    char coin_acqmode[COINC_PKT_PER_BLOCK];
    uint16_t coin_pktNum[COINC_PKT_PER_BLOCK];
    uint16_t coin_modNum[COINC_PKT_PER_BLOCK];
    uint8_t coin_quaNum[COINC_PKT_PER_BLOCK];
    uint32_t coin_pktUTC[COINC_PKT_PER_BLOCK];
    uint32_t coin_pktNSEC[COINC_PKT_PER_BLOCK];
    long int coin_tv_sec[COINC_PKT_PER_BLOCK];
    long int coin_tv_usec[COINC_PKT_PER_BLOCK];
    int coinc_block_size;


    int INTSIG;
} HSD_output_block_header_t;

typedef uint8_t HSD_output_header_cache_alignment[
    CACHE_ALIGNMENT - (sizeof(HSD_output_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct HSD_output_block {
    HSD_output_block_header_t header;
    HSD_output_header_cache_alignment padding;  //Maintain cache alignment
    char stream_block[OUTPUTBLOCKSIZE*sizeof(char)];
    char coinc_block[OUTPUTCOICBLOCKSIZE*sizeof(char)];
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

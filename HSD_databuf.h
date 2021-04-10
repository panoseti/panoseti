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
#define IN_PKT_PER_BLOCK        40                      //Number of Pkt stored in each block
#define OUT_MODPAIR_PER_BLOCK   40                      //Max Number of Module Pairs stored in each block
#define COINC_PKT_PER_BLOCK     40                      //Max Number of Coinc packets stored in each block

//Defining Imaging Data Values
#define QUABOPERMODULE          4
#define PKTPERPAIR              QUABOPERMODULE*2
#define SCIDATASIZE             256
#define MODPAIRDATASIZE         PKTPERPAIR*PKTDATASIZE

//Defining the Block Sizes for the Input and Ouput Buffers
#define INPUTBLOCKSIZE          IN_PKT_PER_BLOCK*PKTDATASIZE                    //Input Block size includes headers
#define OUTPUTBLOCKSIZE         OUT_MODPAIR_PER_BLOCK*MODPAIRDATASIZE           //Output Stream Block size excludes headers
#define OUTPUTCOICBLOCKSIZE     COINC_PKT_PER_BLOCK*PKTDATASIZE                 //Output Coinc Block size excluding headers


#define BLOCKSIZE           INPUTBLOCKSIZE


//Definng the numerical values
#define RANK                    2
#define HKDATASIZE              464
#define DATABLOCKSIZE           SCIDATASIZE*PKTPERPAIR+64+16
#define HKFIELDS                27
#define GPSFIELDS               9
#define NANOSECTHRESHOLD        20
#define MODULEINDEXSIZE         0xffff


#define MODULEPAIR_FORMAT "ModulePair_%05u_%05u"

//Defining the string buffer size
#define STRBUFFSIZE 80



/**
 * The Housekeeping packet structure used to write to HDF5 table
 */
typedef struct HKPackets {
    char SYSTIME[STRBUFFSIZE];
    uint16_t BOARDLOC;
    float HVMON0, HVMON1, HVMON2, HVMON3;
    float HVIMON0, HVIMON1, HVIMON2, HVIMON3;
    float RAWHVMON;
    float V12MON, V18MON, V33MON, V37MON;
    float I10MON, I18MON, I33MON;
    float TEMP1;
    float TEMP2;
    float VCCINT, VCCAUX;
    uint64_t UID;
    uint8_t SHUTTER_STATUS, LIGHT_STATUS;
    uint32_t FWID0, FWID1;
} HKPackets_t;

//Defining other dependencies needed for writing to HDF5 tables
const HKPackets_t  HK_dst_buf[0] = {};

const size_t HK_dst_size = sizeof(HKPackets_t);

const size_t HK_dst_offset[HKFIELDS] = { HOFFSET( HKPackets_t, SYSTIME ),
                                        HOFFSET( HKPackets_t, BOARDLOC),
                                        HOFFSET( HKPackets_t, HVMON0 ),
                                        HOFFSET( HKPackets_t, HVMON1 ),
                                        HOFFSET( HKPackets_t, HVMON2 ),
                                        HOFFSET( HKPackets_t, HVMON3 ),
                                        HOFFSET( HKPackets_t, HVIMON0 ),
                                        HOFFSET( HKPackets_t, HVIMON1 ),
                                        HOFFSET( HKPackets_t, HVIMON2 ),
                                        HOFFSET( HKPackets_t, HVIMON3 ),
                                        HOFFSET( HKPackets_t, RAWHVMON ),
                                        HOFFSET( HKPackets_t, V12MON ),
                                        HOFFSET( HKPackets_t, V18MON ),
                                        HOFFSET( HKPackets_t, V33MON ),
                                        HOFFSET( HKPackets_t, V37MON ),
                                        HOFFSET( HKPackets_t, I10MON ),
                                        HOFFSET( HKPackets_t, I18MON ),
                                        HOFFSET( HKPackets_t, I33MON ),
                                        HOFFSET( HKPackets_t, TEMP1 ),
                                        HOFFSET( HKPackets_t, TEMP2 ),
                                        HOFFSET( HKPackets_t, VCCINT ),
                                        HOFFSET( HKPackets_t, VCCAUX ),
                                        HOFFSET( HKPackets_t, UID ),
                                        HOFFSET( HKPackets_t, SHUTTER_STATUS ),
                                        HOFFSET( HKPackets_t, LIGHT_STATUS ),
                                        HOFFSET( HKPackets_t, FWID0 ),
                                        HOFFSET( HKPackets_t, FWID1 )};

const size_t HK_dst_sizes[HKFIELDS] = { sizeof( HK_dst_buf[0].SYSTIME),
                                        sizeof( HK_dst_buf[0].BOARDLOC),
                                        sizeof( HK_dst_buf[0].HVMON0),
                                        sizeof( HK_dst_buf[0].HVMON1),
                                        sizeof( HK_dst_buf[0].HVMON2),
                                        sizeof( HK_dst_buf[0].HVMON3),
                                        sizeof( HK_dst_buf[0].HVIMON0),
                                        sizeof( HK_dst_buf[0].HVIMON1),
                                        sizeof( HK_dst_buf[0].HVIMON2),
                                        sizeof( HK_dst_buf[0].HVIMON3),
                                        sizeof( HK_dst_buf[0].RAWHVMON),
                                        sizeof( HK_dst_buf[0].V12MON),
                                        sizeof( HK_dst_buf[0].V18MON),
                                        sizeof( HK_dst_buf[0].V33MON),
                                        sizeof( HK_dst_buf[0].V37MON),
                                        sizeof( HK_dst_buf[0].I10MON),
                                        sizeof( HK_dst_buf[0].I18MON),
                                        sizeof( HK_dst_buf[0].I33MON),
                                        sizeof( HK_dst_buf[0].TEMP1),
                                        sizeof( HK_dst_buf[0].TEMP2),
                                        sizeof( HK_dst_buf[0].VCCINT),
                                        sizeof( HK_dst_buf[0].VCCAUX),
                                        sizeof( HK_dst_buf[0].UID),
                                        sizeof( HK_dst_buf[0].SHUTTER_STATUS),
                                        sizeof( HK_dst_buf[0].LIGHT_STATUS),
                                        sizeof( HK_dst_buf[0].FWID0),
                                        sizeof( HK_dst_buf[0].FWID1)};

const char *HK_field_names[HKFIELDS] = { "SYSTIME", "BOARDLOC",
                                        "HVMON0","HVMON1","HVMON2","HVMON3",
                                        "HVIMON0","HVIMON1","HVIMON2","HVIMON3",
                                        "RAWHVMON",
                                        "V12MON","V18MON","V33MON","V37MON",
                                        "I10MON","I18MON","I33MON",
                                        "TEMP1","TEMP2",
                                        "VCCINT","VCCAUX",
                                        "UID",
                                        "SHUTTER_STATUS","LIGHT_SENSOR_STATUS",
                                        "FWID0","FWID1"};

/**
 * Getting the string type for the HK_fields_type
 */
hid_t get_H5T_string_type(){
    hid_t string_type;

    string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size( string_type, STRBUFFSIZE );
    return string_type;
}

const hid_t HK_field_types[HKFIELDS] = { get_H5T_string_type(), H5T_STD_U16LE,                              // SYSTIME, BOARDLOC
                                H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,     // HVMON0-3
                                H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,     // HVIMON0-3
                                H5T_NATIVE_FLOAT,                                                           // RAWHVMON
                                H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,     // V12MON, V18MON, V33MON, V37MON           
                                H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,                       // I10MON, I18MON, I33MON        
                                H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,                                         // TEMP1, TEMP2                        
                                H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,                                         // VCCINT, VCCAUX              
                                H5T_STD_U64LE,                                                              // UID
                                H5T_STD_I8LE,H5T_STD_I8LE,                                                  // SHUTTER and LIGHT_SENSOR STATUS
                                H5T_STD_U32LE,H5T_STD_U32LE                                                 // FWID0 and FWID1
};

/**
 * The GPS packet structure used for writing to HDF5 table
 */
typedef struct GPSPackets {
    char GPSTIME[STRBUFFSIZE];
    uint32_t TOW;
    uint16_t WEEKNUMBER;
    uint8_t UTCOFFSET;
    char TIMEFLAG[STRBUFFSIZE];
    char PPSFLAG[STRBUFFSIZE];
    uint8_t TIMESET;
    uint8_t UTCINFO;
    uint8_t TIMEFROMGPS;
} GPSPackets_t;

//Defining other depenedencies for writing GPS data to HDF5 table
const GPSPackets_t  GPS_dst_buf[0] = {};

const size_t GPS_dst_size = sizeof(GPSPackets_t);

const size_t GPS_dst_offset[GPSFIELDS] = { HOFFSET( GPSPackets_t, GPSTIME ),
                                        HOFFSET( GPSPackets_t, TOW),
                                        HOFFSET( GPSPackets_t, WEEKNUMBER),
                                        HOFFSET( GPSPackets_t, UTCOFFSET),
                                        HOFFSET( GPSPackets_t, TIMEFLAG),
                                        HOFFSET( GPSPackets_t, PPSFLAG),
                                        HOFFSET( GPSPackets_t, TIMESET),
                                        HOFFSET( GPSPackets_t, UTCINFO),
                                        HOFFSET( GPSPackets_t, TIMEFROMGPS)};

const size_t GPS_dst_sizes[GPSFIELDS] = { sizeof( GPS_dst_buf[0].GPSTIME),
                                        sizeof( GPS_dst_buf[0].TOW),
                                        sizeof( GPS_dst_buf[0].WEEKNUMBER),
                                        sizeof( GPS_dst_buf[0].UTCOFFSET),
                                        sizeof( GPS_dst_buf[0].TIMEFLAG),
                                        sizeof( GPS_dst_buf[0].PPSFLAG),
                                        sizeof( GPS_dst_buf[0].TIMESET),
                                        sizeof( GPS_dst_buf[0].UTCINFO),
                                        sizeof( GPS_dst_buf[0].TIMEFROMGPS)};

const char *GPS_field_names[GPSFIELDS] = { "GPSTIME",
                                        "TOW",
                                        "WEEKNUMBER",
                                        "UTCOFFSET",
                                        "TIMEFLAG",
                                        "PPSFLAG",
                                        "TIMESET",
                                        "UTCINFO",
                                        "TIMEFROMGPS"};

const hid_t GPS_field_types[GPSFIELDS] = { get_H5T_string_type(),   // GPSTIME
                                        H5T_STD_U32LE,              // TOW;
                                        H5T_STD_U16LE,              // WEEKNUMBER
                                        H5T_STD_U8LE,               // UTCOFFSET
                                        get_H5T_string_type(),      // TIMEFLAG[STRBUFFSIZE]
                                        get_H5T_string_type(),      // PPSFLAG[STRBUFFSIZE]
                                        H5T_STD_U8LE,               // TIMESET
                                        H5T_STD_U8LE,               // UTCINFO
                                        H5T_STD_U8LE                // TIMEFROMGPS
};

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
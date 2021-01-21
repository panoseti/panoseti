/*
 * demo1_output_thread.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include "hashpipe.h"
#include "HSD_databuf.h"
#include "hiredis/hiredis.h"
#include "hdf5.h"
#include "hdf5_hl.h"

//Defining the names of redis keys and files
#define OBSERVATORY "LICK"
#define GPSPRIMNAME "GPSPRIM"
#define GPSSUPPNAME "GPSSUPP"
#define WRSWITCHNAME "WRSWITCH"
#define CONFIGFILE "./modulePair.config"

//Defining the Formats that will be used within the HDF5 data file
#define H5FILE_NAME_FORMAT "PANOSETI_%s_%04i_%02i_%02i_%02i-%02i-%02i.h5"
#define TIME_FORMAT "%04i-%02i-%02iT%02i:%02i:%02i UTC"
#define FRAME_FORMAT "Frame%05i"
#define IMGDATA_FORMAT "DATA%09i"
#define PHDATA_FORMAT "PH_Module%05i_Quabo%01i_UTC%09i_NANOSEC%09i_PKTNUM%05i"
#define QUABO_FORMAT "QUABO%05i_%01i"
#define HK_TABLENAME_FORAMT "HK_Module%05i_Quabo%01i"
#define HK_TABLETITLE_FORMAT "HouseKeeping Data for Module%05i_Quabo%01i"
#define MODULEPAIR_FORMAT "ModulePair_%05u_%05u"

//Definng the numerical values
#define RANK 2
#define QUABOPERMODULE 4
#define PKTPERPAIR QUABOPERMODULE*2
#define SCIDATASIZE 256
#define HKDATASIZE 464
#define DATABLOCKSIZE SCIDATASIZE*PKTPERPAIR+64+16
#define HKFIELDS 27
#define GPSFIELDS 9
#define NANOSECTHRESHOLD 20

//Defining the string buffer size
#define STRBUFFSIZE 50

//Defining the static values for the storage values for HDF5 file
static hsize_t storageDim[RANK] = {PKTPERPAIR,SCIDATASIZE};

static hid_t storageSpace = H5Screate_simple(RANK, storageDim, NULL);

static hid_t storageTypebit16 = H5Tcopy(H5T_STD_U16LE);

static hid_t storageTypebit8 = H5Tcopy(H5T_STD_U8LE);

static long long fileSize = 0;

static long long maxFileSize = 0; //IN UNITS OF APPROX 2 BYTES OR 16 bits

/**
 * The fileID structure for the current HDF5 opened.
 */
typedef struct fileIDs {
    hid_t       file;         /* file and dataset handles */
    hid_t       bit16IMGData, bit8IMGData, PHData, ShortTransient, bit16HCData, bit8HCData, DynamicMeta, StaticMeta;
} fileIDs_t;

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

/**
 * The module ID structure that is used to store a lot of the information regarding the current pair of module.
 * Module pairs consists of PKTPERPAIR(8) quabos.
 */
typedef struct modulePairData {
    hid_t ID16bit;
    hid_t ID8bit;
    hid_t dynamicMeta;
    uint8_t status;   // Determine the which part of the data is filled 0:neither filled 1:First rank filled 2: Second rank filled
    int lastMode;
    int mod1Name;
    int mod2Name;
    uint8_t data[PKTPERPAIR*SCIDATASIZE*2];
    uint16_t PKTNUM[PKTPERPAIR];
    long int tv_sec[PKTPERPAIR];
    long int tv_usec[PKTPERPAIR];
    uint32_t NANOSEC[PKTPERPAIR];
    uint32_t upperNANOSEC;
    uint32_t lowerNANOSEC;
    int bit16dataNum;
    int bit8dataNum;
    modulePairData* next_moduleID;
} modulePairData_t;

/**
 * Creating a new module ID object given the ID values and module numbers.
 */
modulePairData_t* modulePairData_t_new(hid_t ID16, hid_t ID8, hid_t dynamicMD, unsigned int mod1, unsigned int mod2){
    modulePairData_t* value = (modulePairData_t*) malloc(sizeof(struct modulePairData));
    if (value == NULL){
        printf("Error: Unable to malloc space for ModulePairData\n");
        exit(1);
    }
    //moduleFillZeros(value->data, 0);
    value->ID16bit = ID16;
    value->ID8bit = ID8;
    value->dynamicMeta = dynamicMD;
    value->status = 0;
    value->mod1Name = mod1;
    value->mod2Name = mod2;
    value->next_moduleID = NULL;
    value->upperNANOSEC = 0;
    value->lowerNANOSEC = 0;
    value->bit16dataNum = 0;
    value->bit8dataNum = 0;
    return value;
}

/**
 * Creating a new module ID with zeroed/null values
 */
modulePairData_t* modulePairData_t_new(){
    return modulePairData_t_new(0,0,0,-1,-1);
}

/**
 * Creating a new module ID with only 1 module for when new module is detected and not in module pair config file
 */
modulePairData_t* modulePairData_t_new(hid_t ID16, hid_t ID8, hid_t dynamicMD, unsigned mod1){
    return modulePairData_t_new(ID16, ID8, dynamicMD, mod1, -1);
}

/**
 * Getting the module ID from the linked list
 */
modulePairData_t* get_moduleID(modulePairData_t* list, unsigned int ind){
    if(list != NULL && ind > 0)
        return get_moduleID(list->next_moduleID, ind-1);
    return list;
}

/**
 * Filling the module ID with Zeros
 */
void moduleFillZeros(modulePairData_t* module, uint8_t status){
    for(int i = 0; i < PKTPERPAIR; i++){
        if(!((status >> i) & 0x01)){
            if (module->lastMode == 16){
                memset(module->data + (i*SCIDATASIZE*2), 0, SCIDATASIZE*2*sizeof(uint8_t));
            } else if (module->lastMode == 8) {
                memset(module->data + (i*SCIDATASIZE), 0, SCIDATASIZE*sizeof(uint8_t));
            }
            module->PKTNUM[i] = 0;
            module->tv_sec[i] = 0;
            module->tv_usec[i] = 0;
            module->NANOSEC[i] = 0;
        }
    }
}

/**
 * Create a singular string attribute attached to the given group.
 */
void createStrAttribute(hid_t group, const char* name, char* data) {
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;

    dataspace = H5Screate(H5S_SCALAR);
    if (dataspace < 0){
        printf("Error: Unable to create HDF5 dataspace for string attribute - %s.", name);
        return;
    }

    datatype = H5Tcopy(H5T_C_S1);
    if (datatype < 0 ){
        printf("Error: Unable to create HDF5 datatype for string attribute - %s.", name);
        return;
    }
    H5Tset_size(datatype, strlen(data));
    H5Tset_strpad(datatype, H5T_STR_NULLTERM);
    H5Tset_cset(datatype, H5T_CSET_UTF8);

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute < 0 ){
        printf("Error: Unable to create HDF5 Attribute for string attribute - %s.", name);
        return;
    }

    if (H5Awrite(attribute, datatype, data) < 0){
        printf("Warning: Unable to write HDF5 Attribute for string attribute - %s.", name);
    }
    // Add size to fileSize
    fileSize += STRBUFFSIZE;

    if (H5Sclose(dataspace) < 0){
        printf("Warning: Unable to close HDF5 dataspace for string attribute - %s", name);
    }
    if (H5Tclose(datatype) < 0){
        printf("Warning: Unable to close HDF5 datatype for string attribute - %s", name);
    }
    if (H5Aclose(attribute) < 0){
        printf("Warning: Unable to close HDF5 attribute for string attribute - %s", name);
    }

}

/**
 * Create a multidimensional string attribute attached to the given group.
 */
void createStrAttribute2(hid_t group, const char* name, hsize_t* dimsf, char data[PKTPERPAIR][STRBUFFSIZE]) {
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;

    dataspace = H5Screate_simple(sizeof(dimsf)/sizeof(dimsf[0]), dimsf, NULL);
    if (dataspace < 0){
        printf("Error: Unable to create HDF5 dataspace for string attribute - %s.", name);
        return;
    }

    datatype = H5Tcopy(H5T_C_S1);
    if (datatype < 0 ){
        printf("Error: Unable to create HDF5 datatype for string attribute - %s.", name);
        return;
    }
    H5Tset_size(datatype, STRBUFFSIZE);
    H5Tset_strpad(datatype, H5T_STR_NULLTERM);
    H5Tset_cset(datatype, H5T_CSET_UTF8);

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute < 0 ){
        printf("Error: Unable to create HDF5 Attribute for string attribute - %s.", name);
        return;
    }
    
    if (H5Awrite(attribute, datatype, data[0]) < 0){
        printf("Warning: Unable to write HDF5 Attribute for string attribute - %s.", name);
    }
    fileSize += STRBUFFSIZE;

    if (H5Sclose(dataspace) < 0){
        printf("Warning: Unable to close HDF5 dataspace for string attribute - %s", name);
    }
    if (H5Tclose(datatype) < 0){
        printf("Warning: Unable to close HDF5 datatype for string attribute - %s", name);
    }
    if (H5Aclose(attribute) < 0){
        printf("Warning: Unable to close HDF5 attribute for string attribute - %s", name);
    }
}

/**
 * Create a singular numerical attribute attached to the given group
 */
void createNumAttribute(hid_t group, const char* name, hid_t dtype, unsigned long long data) {
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;
    unsigned long long attr_data[1];
    attr_data[0] = data;

    
    dataspace = H5Screate(H5S_SCALAR);
    if (dataspace < 0){
        printf("Error: Unable to create HDF5 dataspace for numerical attribute - %s.", name);
        return;
    }

    datatype = H5Tcopy(dtype);
    if (datatype < 0 ){
        printf("Error: Unable to create HDF5 datatype for numerical attribute - %s.", name);
        return;
    }

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute < 0 ){
        printf("Error: Unable to create HDF5 Attribute for numerical attribute - %s.", name);
        return;
    }

    if (H5Awrite(attribute, dtype, attr_data) < 0){
        printf("Warning: Unable to write HDF5 Attribute for numerical attribute - %s.", name);
    }
    fileSize += 16;
    
    if (H5Sclose(dataspace) < 0){
        printf("Warning: Unable to close HDF5 dataspace for numerical attribute - %s", name);
    }
    if (H5Tclose(datatype) < 0){
        printf("Warning: Unable to close HDF5 datatype for numerical attribute - %s", name);
    }
    if (H5Aclose(attribute) < 0){
        printf("Warning: Unable to close HDF5 attribute for numerical attribute - %s", name);
    }
}

/**
 * Create a multidensional numberical attribute attached to the given group
 */
void createNumAttribute2(hid_t group, const char* name, hid_t dtype, hsize_t* dimsf, void* data) {
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;

    dataspace = H5Screate_simple(sizeof(dimsf)/sizeof(dimsf[0]), dimsf, NULL);
    if (dataspace < 0){
        printf("Error: Unable to create HDF5 dataspace for numerical attribute - %s.", name);
        return;
    }

    datatype = H5Tcopy(dtype);
    if (datatype < 0 ){
        printf("Error: Unable to create HDF5 datatype for numerical attribute - %s.", name);
        return;
    }

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute < 0 ){
        printf("Error: Unable to create HDF5 Attribute for numerical attribute - %s.", name);
        return;
    }

    if (H5Awrite(attribute, dtype, data) < 0){
        printf("Warning: Unable to write HDF5 Attribute for numerical attribute - %s.", name);
    }
    fileSize += 32;

    if (H5Sclose(dataspace) < 0){
        printf("Warning: Unable to close HDF5 dataspace for numerical attribute - %s.", name);
    }
    if (H5Tclose(datatype) < 0){
        printf("Warning: Unable to close HDF5 datatype for numerical attribute - %s.", name);
    }
    if (H5Aclose(attribute) < 0){
        printf("Warning: Unable to close HDF5 attribute for numerical attribute - %s.", name);
    }
}

/**
 * Create a singular float attribute attached to the given group
 */
void createFloatAttribute(hid_t group, const char* name, float data){
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;
    float attr_data[1];
    attr_data[0] = data;

    dataspace = H5Screate(H5S_SCALAR);
    if (dataspace < 0){
        printf("Error: Unable to create HDF5 dataspace for floating point attribute - %s.", name);
        return;
    }

    datatype = H5Tcopy(H5T_NATIVE_FLOAT);
    if (datatype < 0 ){
        printf("Error: Unable to create HDF5 datatype for floating point attribute - %s.", name);
        return;
    }

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute < 0 ){
        printf("Error: Unable to create HDF5 Attribute for floating point attribute - %s.", name);
        return;
    }

    if (H5Awrite(attribute, H5T_NATIVE_FLOAT, attr_data) < 0){
        printf("Warning: Unable to write HDF5 Attribute for floating point attribute - %s.", name);
    }
    fileSize += 32;
    
    if (H5Sclose(dataspace) < 0){
        printf("Warning: Unable to close HDF5 dataspace for floating point attribute - %s.", name);
    }
    if (H5Tclose(datatype) < 0){
        printf("Warning: Unable to close HDF5 datatype for floating point attribute - %s.", name);
    }
    if (H5Aclose(attribute) < 0){
        printf("Warning: Unable to close HDF5 attribute for floating point attribute - %s.", name);
    }
}

/**
 * Create a singular double attribute attached to the given group.
 */
void createDoubleAttribute(hid_t group, const char* name, double data){
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;
    double attr_data[1];
    attr_data[0] = data;

    dataspace = H5Screate(H5S_SCALAR);
    if (dataspace < 0){
        printf("Error: Unable to create HDF5 dataspace for double precision attribute - %s.", name);
        return;
    }

    datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
    if (datatype < 0 ){
        printf("Error: Unable to create HDF5 datatype for double precision attribute - %s.", name);
        return;
    }

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute < 0 ){
        printf("Error: Unable to create HDF5 Attribute for double precision attribute - %s.", name);
        return;
    }

    if (H5Awrite(attribute, H5T_NATIVE_DOUBLE, attr_data) < 0){
        printf("Warning: Unable to write HDF5 Attribute for double precision attribute - %s.", name);
    }
    fileSize += 64;
    
    if (H5Sclose(dataspace) < 0){
        printf("Warning: Unable to close HDF5 dataspace for double precision attribute - %s.", name);
    }
    if (H5Tclose(datatype) < 0){
        printf("Warning: Unable to close HDF5 datatype for double precision attribute - %s.", name);
    }
    if (H5Aclose(attribute) < 0){
        printf("Warning: Unable to close HDF5 attribute for double precision attribute - %s.", name);
    }
}

/**
 * Create a module pair within the HDF5 file located at the group.
 */
hid_t createModPair(hid_t group, unsigned int mod1Name, unsigned int mod2Name) {
    hid_t   modulePair;
    char    modName[STRBUFFSIZE];
    hsize_t dimsf[1];
    uint64_t modNames[2];
    modNames[0] = mod1Name;
    modNames[1] = mod2Name;
    dimsf[0] = 2;
    sprintf(modName, MODULEPAIR_FORMAT, mod1Name, mod2Name);

    modulePair = H5Gcreate(group, modName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if(modulePair < 0){
        printf("Error: Unable to create module Pair cache for module %u, %u.", mod1Name, mod2Name);
        exit(1);
    }

    createNumAttribute2(modulePair, "ModuleNum", H5T_STD_U64LE, dimsf, modNames);

    return modulePair;//H5Gclose(modulePair);

}

/**
 * Create a singular module within the HDF5 file located at the group.
 */
hid_t createMod(hid_t group, unsigned int mod1Name){
    hid_t   modulePair;
    char    modName[STRBUFFSIZE];

    sprintf(modName, "./ModulePair_%05u", mod1Name);

    modulePair = H5Gcreate(group, modName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    createNumAttribute(modulePair, "ModuleNum", H5T_STD_U64LE , mod1Name);

    return modulePair;
}

/**
 * Create new quabo tables within the HDF5 file located at the group.
 */
void createQuaboTables(hid_t group, modulePairData_t* module){

    HKPackets_t HK_data;
    char tableName[50];
    char tableTitle[50];
    for (int i = 0; i < QUABOPERMODULE; i++) {
        sprintf(tableName, HK_TABLENAME_FORAMT, module->mod1Name, i);
        sprintf(tableTitle, HK_TABLETITLE_FORMAT, module->mod1Name, i);

        if (H5TBmake_table(tableTitle, group, tableName, HKFIELDS, 0,
                            HK_dst_size, HK_field_names, HK_dst_offset, HK_field_types,
                            100, NULL, 0, &HK_data) < 0){
                                printf("Error: Unable to create quabo tables in HDF5 file.\n");
                                exit(1);
                            }
    }

    for (int i = 0; i < QUABOPERMODULE; i++) {
        sprintf(tableName, HK_TABLENAME_FORAMT, module->mod2Name, i);
        sprintf(tableTitle, HK_TABLETITLE_FORMAT, module->mod2Name, i);

        if (H5TBmake_table(tableTitle, group, tableName, HKFIELDS, 0,
                            HK_dst_size, HK_field_names, HK_dst_offset, HK_field_types,
                            100, NULL, 0, &HK_data) < 0){
                                printf("Error: Unable to create quabo tables in HDF5 file.\n");
                                exit(1);
                            }
    }
}

/**
 * Create new GPS tables within the HDF5 files located at the group
 */
void createGPSTable(hid_t group){
    GPSPackets_t GPS_data;

    if (H5TBmake_table(GPSPRIMNAME, group, GPSPRIMNAME, GPSFIELDS, 0,
                            GPS_dst_size, GPS_field_names, GPS_dst_offset, GPS_field_types,
                            100, NULL, 0, &GPS_data) < 0){
                                printf("Unable to create GPS Table for HDF5 file\n");
                                exit(1);
                            }
}

/**
 * Create new White Rabbit Switch tables within the HDF5 files located at the group.
 */
void createWRTable(){

}

/**
 * Inialize the metadata resources such as GPS and WR tables.
 */
void createDMetaResources(hid_t group){
    createGPSTable(group);
    createWRTable();
}

/**
 * Initialize a new file given a name and time.
 */
fileIDs_t* createNewFile(char* fileName, char* currTime){
    fileIDs_t* newfile = (fileIDs_t*) malloc(sizeof(struct fileIDs));
    if(newfile == NULL){
        printf("Error: Unable to malloc space for new file.\n");
        exit(1);
    }

    newfile->file = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    createStrAttribute(newfile->file, "dateCreated", currTime);
    newfile->bit16IMGData = H5Gcreate(newfile->file, "/bit16IMGData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->bit8IMGData = H5Gcreate(newfile->file, "/bit8IMGData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->PHData = H5Gcreate(newfile->file, "/PHData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->ShortTransient = H5Gcreate(newfile->file, "/ShortTransient", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->bit16HCData =  H5Gcreate(newfile->file, "/bit16HCData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->bit8HCData = H5Gcreate(newfile->file, "/bit8HCData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    newfile->DynamicMeta = H5Gcreate(newfile->file, "/DynamicMeta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->StaticMeta = H5Gcreate(newfile->file, "/StaticMeta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if(newfile->file < 0 || newfile->bit16IMGData < 0 || newfile->bit8IMGData < 0 ||
        newfile->PHData < 0 || newfile->ShortTransient < 0 || newfile->bit16HCData < 0 ||
        newfile->bit8HCData < 0 || newfile->DynamicMeta < 0 || newfile->StaticMeta < 0){
            printf("Error in creating HD5f file\n");
            exit(1);
    } else {
        printf("Created new file: %s\n", fileName);
    }

    return newfile;
}

/**
 * Send Redis Command
 * @return The redis reply from HIREDIS
 */
redisReply* sendHSETRedisCommand(redisContext* redisServer, const char* HSETName, const char* Value){
    redisReply* reply = (redisReply *)redisCommand(redisServer, "HGET %s %s", HSETName, Value);
    if (reply->type != REDIS_REPLY_STATUS){
        printf("Warning: Redis was unable to get replay with the command - HGET %s %s\n", HSETName, Value);
        freeReplyObject(reply);
        return NULL;
    }
    return reply;
}

/**
 * Send Redis Command
 * @return The redis reply from HIREDIS
 */
redisReply* sendHSETRedisCommand(redisContext* redisServer, int HSETName, const char* Value){
    redisReply* reply = (redisReply *)redisCommand(redisServer, "HGET %i %s", HSETName, Value);
    if (reply->type != REDIS_REPLY_STATUS){
        printf("Warning: Redis was unable to get replay with the command - HGET %i %s\n", HSETName, Value);
        freeReplyObject(reply);
        return NULL;
    }
    return reply;
}

/**
 * Fetch the Housekeeping data from the Redis database for the given boardloc or quabo id.
 */
void fetchHKdata(HKPackets_t* HK, uint16_t BOARDLOC, redisContext* redisServer) {
    redisReply *reply;

    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "SYSTIME");
    if (reply != NULL){
        strcpy(HK->SYSTIME, reply->str);
        freeReplyObject(reply);
    } else {strcpy(HK->SYSTIME, "");}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "BOARDLOC");
    if (reply != NULL){
        HK->BOARDLOC = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    } else {HK->BOARDLOC = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVMON0");
    if (reply != NULL){
        HK->HVMON0 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->HVMON0 = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVMON1");
    if (reply != NULL){
        HK->HVMON1 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->HVMON1 = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVMON2");
    if (reply != NULL){
        HK->HVMON2 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->HVMON2 = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVMON3");
    if (reply != NULL){
        HK->HVMON3 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->HVMON3 = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVIMON0");
    if (reply != NULL){
        HK->HVIMON0 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->HVIMON0 = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVIMON1");
    if (reply != NULL){
        HK->HVIMON1 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->HVIMON1 = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVIMON2");
    if (reply != NULL){
        HK->HVIMON2 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->HVIMON2 = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "HVIMON3");
    if (reply != NULL){
        HK->HVIMON3 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->HVIMON3 = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "RAWHVMON");
    if (reply != NULL){
        HK->RAWHVMON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->RAWHVMON = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "V12MON");
    if (reply != NULL){
        HK->V12MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->V12MON = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "V18MON");
    if (reply != NULL){
        HK->V18MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->V18MON = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "V33MON");
    if (reply != NULL){
        HK->V33MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->V33MON = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "V37MON");
    if (reply != NULL){
        HK->V37MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->V37MON = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "I10MON");
    if (reply != NULL){
        HK->I10MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->I10MON = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "I18MON");
    if (reply != NULL){
        HK->I18MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->I18MON = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "I33MON");
    if (reply != NULL){
        HK->I33MON = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->I33MON = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "TEMP1");
    if (reply != NULL){
        HK->TEMP1 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->TEMP1 = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "TEMP2");
    if (reply != NULL){
        HK->TEMP2 = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->TEMP2 = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "VCCINT");
    if (reply != NULL){
        HK->VCCINT = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->VCCINT = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "VCCAUX");
    if (reply != NULL){
        HK->VCCAUX = strtof(reply->str, NULL);
        freeReplyObject(reply);
    } else {HK->VCCAUX = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "UID");
    if (reply != NULL){
        HK->UID = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    } else {HK->UID = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "SHUTTER_STATUS");
    if (reply != NULL){
        HK->SHUTTER_STATUS = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    } else {HK->SHUTTER_STATUS = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "LIGHT_SENSOR_STATUS");
    if (reply != NULL){
        HK->LIGHT_STATUS = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    } else {HK->LIGHT_STATUS = 0;}
    
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "FWID0");
    if (reply != NULL){
        HK->FWID0 = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    } else {HK->FWID0 = 0;}
    reply = sendHSETRedisCommand(redisServer, BOARDLOC, "FWID1");
    if (reply != NULL){
        HK->FWID1 = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    } else {HK->FWID1 = 0;}
}

/**
 * Get the GPS data from the Redis Server.
 */
void fetchGPSdata(GPSPackets_t* GPS, redisContext* redisServer) {
    redisReply *reply;

    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "GPSTIME");
    if (reply != NULL){
        strcpy(GPS->GPSTIME, reply->str);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "TOW");
    if (reply != NULL){
        GPS->TOW = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "WEEKNUMBER");
    if (reply != NULL){
        GPS->WEEKNUMBER = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "UTCOFFSET");
    if (reply != NULL){
        GPS->UTCOFFSET = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "TIMEFLAG");
    if (reply != NULL){
        strcpy(GPS->TIMEFLAG, reply->str);
        freeReplyObject(reply);
    }
    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "PPSFLAG");
    if (reply != NULL){
        strcpy(GPS->PPSFLAG, reply->str);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "TIMESET");
    if (reply != NULL){
        GPS->TIMESET = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "UTCINFO");
    if (reply != NULL){
        GPS->UTCINFO = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
    reply = sendHSETRedisCommand(redisServer, GPSPRIMNAME, "TIMEFROMGPS");
    if (reply != NULL){
        GPS->TIMEFROMGPS = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
}

/**
 * Check if housekeeping data for the module pair has been updated and if so get and store it in the HDF5 file.
 */
void check_storeHK(redisContext* redisServer, modulePairData_t* modHead){
    HKPackets_t* HKdata = (HKPackets_t *)malloc(sizeof(HKPackets));
    if (HKdata == NULL){
        printf("Error: Unable to malloc space for House Keeping Object.\n");
        exit(1);
    }
    modulePairData_t* currentMod;
    redisReply* reply;
    uint16_t BOARDLOC;
    char tableName[50];

    currentMod = modHead;
    
    while(currentMod != NULL){
 
        //Updating all the Quabos from Module 1
        BOARDLOC = (currentMod->mod1Name << 2) & 0xfffc;
        
        for(int i = 0; i < 4; i++){
            reply = (redisReply *)redisCommand(redisServer, "HGET UPDATED %u", BOARDLOC);

            if (strtol(reply->str, NULL, 10)){
                freeReplyObject(reply);

                fetchHKdata(HKdata, BOARDLOC, redisServer);
                sprintf(tableName, HK_TABLENAME_FORAMT, currentMod->mod1Name, i);
                if (H5TBappend_records(currentMod->dynamicMeta, tableName, 1, HK_dst_size, HK_dst_offset, HK_dst_sizes, HKdata) < 0){
                    printf("Warning: Unable to write HK Data for module %i-%i", currentMod->mod1Name, i);
                }
                fileSize += HKDATASIZE;

                reply = (redisReply *)redisCommand(redisServer, "HSET UPDATED %u 0", BOARDLOC);
            }

            freeReplyObject(reply);
            BOARDLOC++;
        }

        if(currentMod->mod2Name != -1){
            //Updating all the Quabos from Module 2
            BOARDLOC = (currentMod->mod2Name << 2) & 0xfffc;

            for(int i = 0; i < 4; i++){
                reply = (redisReply *)redisCommand(redisServer, "HGET UPDATED %u", BOARDLOC);

                if (strtol(reply->str, NULL, 10)){
                    freeReplyObject(reply);

                    fetchHKdata(HKdata, BOARDLOC, redisServer);
                    sprintf(tableName, HK_TABLENAME_FORAMT, currentMod->mod2Name, i);
                    if (H5TBappend_records(currentMod->dynamicMeta, tableName, 1, HK_dst_size, HK_dst_offset, HK_dst_sizes, HKdata) < 0){
                        printf("Warning: Unable to write HK Data for module %i-%i", currentMod->mod1Name, i);
                    }

                    reply = (redisReply *)redisCommand(redisServer, "HSET UPDATED %u 0", BOARDLOC);
                }

                freeReplyObject(reply);
                BOARDLOC++;
            }
        }

        //Update to Next Module
        currentMod = currentMod->next_moduleID;
    }

    free(HKdata);
}

/**
 * Check if the GPS Primary data have been updated and if so store the GPS Primary data in the HDF5 file.
 */
void check_storeGPS(redisContext* redisServer, hid_t group){
    GPSPackets_t* GPSdata = (GPSPackets_t *)malloc(sizeof(GPSPackets));
    if (GPSdata == NULL){
        printf("Warning: Unable to malloc space for GPS Object. Skipping GPS Data storage\n");
        return;
    }
    redisReply* reply = (redisReply *)redisCommand(redisServer, "HGET UPDATED %s", GPSPRIMNAME);
    if (reply->type != REDIS_REPLY_STATUS){
        printf("Warning: Unable to get GPS's UPDATED Flag from Redis. Skipping GPS Data.\n");
        return;
    }

    if(strtol(reply->str, NULL, 10)){
        freeReplyObject(reply);

        fetchGPSdata(GPSdata, redisServer);
        if (H5TBappend_records(group, GPSPRIMNAME, 1, GPS_dst_size, GPS_dst_offset, GPS_dst_sizes, GPSdata) < 0){
            printf("Warning: Unable to append GPS data to table in HDF5 file.\n");
        }

        reply = (redisReply *)redisCommand(redisServer, "HSET UPDATED %s 0", GPSPRIMNAME);

        if (reply->type != REDIS_REPLY_STATUS){
            printf("Warning: Unable to set GPS's UPDATED Flag from Redis.\n");
        }
    }

    freeReplyObject(reply);
    free(GPSdata);
}

/**
 * Get and store the GPS Supplimentary data in the HDF5 file.
 */
void get_storeGPSSupp(redisContext* redisServer, hid_t group){
    redisReply* reply;

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "RECEIVERMODE");
    if (reply != NULL){
        createStrAttribute(group, "RECEIVERMODE", reply->str);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DISCIPLININGMODE");
    if (reply != NULL){
        createStrAttribute(group, "DISCIPLININGMODE", reply->str);
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "SELFSURVEYPROGRESS");
    if (reply != NULL){
        createNumAttribute(group, "SELFSURVEYPROGRESS", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "HOLDOVERDURATION");
    if (reply != NULL){
        createNumAttribute(group, "HOLDOVERDURATION", H5T_STD_U32LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DACatRail");
    if (reply != NULL){
        createNumAttribute(group, "DACatRail", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DACnearRail");
    if (reply != NULL){
        createNumAttribute(group, "DACnearRail", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "AntennaOpen");
    if (reply != NULL){
        createNumAttribute(group, "AntennaOpen", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "AntennaShorted");
    if (reply != NULL){
        createNumAttribute(group, "AntennaShorted", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "NotTrackingSatellites");
    if (reply != NULL){
        createNumAttribute(group, "NotTrackingSatellites", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "NotDiscipliningOscillator");
    if (reply != NULL){
        createNumAttribute(group, "NotDiscipliningOscillator", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "SurveyInProgress");
    if (reply != NULL){
        createNumAttribute(group, "SurveyInProgress", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "NoStoredPosition");
    if (reply != NULL){
        createNumAttribute(group, "NoStoredPosition", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "LeapSecondPending");
    if (reply != NULL){
        createNumAttribute(group, "LeapSecondPending", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "InTestMode");
    if (reply != NULL){
        createNumAttribute(group, "InTestMode", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "PositionIsQuestionable");
    if (reply != NULL){
        createNumAttribute(group, "PositionIsQuestionable", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "EEPROMCorrupt");
    if (reply != NULL){
        createNumAttribute(group, "EEPROMCorrupt", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "AlmanacNotComplete");
    if (reply != NULL){
        createNumAttribute(group, "AlmanacNotComplete", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "PPSNotGenerated");
    if (reply != NULL){
        createNumAttribute(group, "PPSNotGenerated", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "GPSDECODINGSTATUS");
    if (reply != NULL){
        createNumAttribute(group, "GPSDECODINGSTATUS", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }
    

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DISCIPLININGACTIVITY");
    if (reply != NULL){
        createNumAttribute(group, "DISCIPLININGACTIVITY", H5T_STD_U8LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "PPSOFFSET");
    if (reply != NULL){
        createFloatAttribute(group, "PPSOFFSET", strtof(reply->str, NULL));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "CLOCKOFFSET");
    if (reply != NULL){
        createFloatAttribute(group, "CLOCKOFFSET", strtof(reply->str, NULL));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DACVALUE");
    if (reply != NULL){
        createNumAttribute(group, "DACVALUE", H5T_STD_U32LE, strtoll(reply->str, NULL, 10));
        freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "DACVOLTAGE");
    if (reply != NULL){
        createFloatAttribute(group, "DACVOLTAGE", strtof(reply->str, NULL));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "TEMPERATURE");
    if (reply != NULL){
        createFloatAttribute(group, "TEMPERATURE", strtof(reply->str, NULL));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "LATITUDE");
    if (reply != NULL){
        createDoubleAttribute(group, "LATITUDE", strtod(reply->str, NULL));
        freeReplyObject(reply);
    }


    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "LONGITUDE");
    if (reply != NULL){
        createDoubleAttribute(group, "LONGITUDE", strtod(reply->str, NULL));
        freeReplyObject(reply);
    }
    
    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "ALTITUDE");
    if (reply != NULL){
        createDoubleAttribute(group, "ALTITUDE", strtod(reply->str, NULL));
            freeReplyObject(reply);
    }

    reply = sendHSETRedisCommand(redisServer, GPSSUPPNAME, "PPSQUANTIZATIONERROR");
    if (reply != NULL){
        createFloatAttribute(group, "PPSQUANTIZATIONERROR", strtof(reply->str, NULL));
        freeReplyObject(reply);
    }
    
}

/**
 * Get and store the White Rabbit Switch data into HDF5 file.
 */
void get_storeWR(redisContext* redisServer, hid_t group){
    redisReply* reply = (redisReply *)redisCommand(redisServer, "HGETALL %s", WRSWITCHNAME);
    if (reply->type != REDIS_REPLY_STATUS){
        printf("Warning: Unable to get WR Swtich Values from Redis. Skipping WR Data.\n");
        return;
    }

    for(int i = 0; i < reply->elements; i=i+2){
        createNumAttribute(group, reply->element[i]->str, H5T_STD_U8LE, strtoll(reply->element[i+1]->str, NULL, 10));
    }
    freeReplyObject(reply);
}

/**
 * Check and store Static data to the HDF5 file.
 */
void getStaticRedisData(redisContext* redisServer, hid_t staticMeta){
    hid_t GPSgroup, WRgroup;
    GPSgroup = H5Gcreate(staticMeta, GPSSUPPNAME, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (GPSgroup < 0){
        printf("Error: Unable to create GPS group in HDF5 file.\n");
        exit(1);
    }
    WRgroup = H5Gcreate(staticMeta, WRSWITCHNAME, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (WRgroup < 0){
        printf("Error: Unable to create White Rabbit group in HDF5 file.\n");
        exit(1);
    }

    get_storeGPSSupp(redisServer, GPSgroup);
    get_storeWR(redisServer, WRgroup);

    if (H5Gclose(GPSgroup) < 0){
        printf("Warning: Unable to close GPS HDF5 Group\n");
    }
    if (H5Gclose(WRgroup) < 0){
        printf("Warning: Unable to close WR HDF5 Group\n");
    }
}

/**
 * Check and store Dynamic data to the HDF5 file.
 */
void getDynamicRedisData(redisContext* redisServer, modulePairData_t* modHead, hid_t dynamicMeta){
    check_storeGPS(redisServer, dynamicMeta);
    check_storeHK(redisServer, modHead);
}

/**
 * Write an PKTPERPAIR frame data block into the module pair in the HDF5 file.
 */
void writeDataBlock(hid_t frame, modulePairData_t* module, int index, int mode){
    hid_t dataset;
    char name[50];
    hsize_t dimsf[1];
    dimsf[0] = PKTPERPAIR;

    sprintf(name, IMGDATA_FORMAT, index);
    
    if(mode == 16){
        dataset = H5Dcreate2(frame, name, storageTypebit16, storageSpace,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset, H5T_STD_U16LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, module->data);
    }else{
        dataset = H5Dcreate2(frame, name, storageTypebit8, storageSpace,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset, H5T_STD_U8LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, module->data);
    }
    

    createNumAttribute2(dataset, "PKTNUM", H5T_STD_U16LE, dimsf, module->PKTNUM);
    //createNumAttribute2(dataset, "UTC", H5T_STD_U32LE, dimsf, module->UTC);
    createNumAttribute2(dataset, "ntp_sec", H5T_NATIVE_LONG, dimsf, module->tv_sec);
    createNumAttribute2(dataset, "ntp_usec", H5T_NATIVE_LONG, dimsf, module->tv_usec);
    createNumAttribute2(dataset, "NANOSEC", H5T_STD_U32LE, dimsf, module->NANOSEC);
    createNumAttribute(dataset, "status", H5T_STD_U8LE, module->status);

    fileSize += DATABLOCKSIZE;

    H5Dclose(dataset);
}

/**
 * Store the data from the data_ptr to the moduleData based on the mode.
 */
void storePktData(uint8_t* moduleData, char* data_ptr, int mode, int quaboIndex){
    uint8_t *data;
    if (mode == 16){
        data = moduleData + (quaboIndex*SCIDATASIZE*2);
        for(int i = 0; i < SCIDATASIZE*2; i++){
            data[i] = data_ptr[i];
        }
    } else if(mode == 8){
        data = moduleData + (quaboIndex*SCIDATASIZE);
        for(int i = 0; i < SCIDATASIZE; i++){
            data[i] = data_ptr[i];
        }
    } else {
        return;
    }
}

//Initializing the linked list to be used for stroing the modulePairData
static modulePairData_t* moduleListBegin = modulePairData_t_new();
static modulePairData_t* moduleListEnd = moduleListBegin;
static modulePairData_t* moduleInd[0xffff] = {NULL};
static fileIDs_t* file;
static redisContext *redisServer;

/**
 * Write the Pulse Height data to disk
 */
void writePHData(uint16_t moduleNum, uint8_t quaboNum, uint16_t PKTNUM, uint32_t UTC, uint32_t NANOSEC, long int tv_sec, long int tv_usec, char* data_ptr){
    hid_t dataset;
    char name[100];
    uint8_t data[SCIDATASIZE*2];
    hsize_t dimsf[1] = {1};

    hsize_t PHDim[1] = {SCIDATASIZE};
    hid_t PHSpace = H5Screate_simple(1, PHDim, NULL);
    hid_t PHType = H5Tcopy(H5T_STD_U16LE);

    sprintf(name, PHDATA_FORMAT, moduleNum, quaboNum, UTC, NANOSEC, PKTNUM);
    //printf("Name = %s\n", name);
    dataset = H5Dcreate2(file->PHData, name, PHType, PHSpace,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    storePktData(data, data_ptr, 16, 0);
    H5Dwrite(dataset, H5T_STD_U16LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    createNumAttribute2(dataset, "PKTNUM", H5T_STD_U16LE, dimsf, &PKTNUM);
    createNumAttribute2(dataset, "UTC", H5T_STD_U32LE, dimsf, &UTC);
    createNumAttribute2(dataset, "ntp_sec", H5T_NATIVE_LONG, dimsf, &tv_sec);
    createNumAttribute2(dataset, "ntp_usec", H5T_NATIVE_LONG, dimsf, &tv_usec);
    createNumAttribute2(dataset, "NANOSEC", H5T_STD_U32LE, dimsf, &NANOSEC);

    fileSize += SCIDATASIZE+80;

    H5Dclose(dataset);
    H5Sclose(PHSpace);
    H5Tclose(PHType);
}

/**
 * Storing the module data to the modulePairData from the data pointer.
 */
void storeData(modulePairData_t* module, char acqmode, uint16_t moduleNum, uint8_t quaboNum, uint16_t PKTNUM, uint32_t UTC, uint32_t NANOSEC, long int tv_sec, long int tv_usec, char* data_ptr){
    //uint16_t* moduleData;
    int mode;
    int quaboIndex;
    uint8_t currentStatus = (0x01 << quaboNum);

    if (acqmode == 0x1){
        writePHData(moduleNum, quaboNum, PKTNUM, UTC, NANOSEC, tv_sec, tv_usec, data_ptr);
        return;
    } else if(acqmode == 0x2 || acqmode == 0x3){
        mode = 16;
    } else if (acqmode == 0x6 || acqmode == 0x7){
        mode = 8;
    } else {
        printf("A new mode was identify acqmode=%X\n", acqmode);
        printf("packet skipped\n");
        return;
    }

    quaboIndex = quaboNum;

    if(moduleNum == module->mod2Name){
        currentStatus = currentStatus << 4;
        quaboIndex += 4;
    }

    if(module->status == 0){
        module->lastMode = mode;
        module->upperNANOSEC = NANOSEC;
        module->lowerNANOSEC = NANOSEC;
    } else if(NANOSEC > module->upperNANOSEC){
        module->upperNANOSEC = NANOSEC;
    } else if (NANOSEC < module->lowerNANOSEC){
        module->lowerNANOSEC = NANOSEC;
    }

    if ((module->status & currentStatus) || module->lastMode != mode || (module->upperNANOSEC - module->lowerNANOSEC) > NANOSECTHRESHOLD){
        //printf("\n");
        moduleFillZeros(module, module->status);
        if (module->lastMode == 16){
            writeDataBlock(module->ID16bit, module, module->bit16dataNum, module->lastMode);
            module->bit16dataNum++;
        }else if (module->lastMode == 8){
            writeDataBlock(module->ID8bit, module, module->bit8dataNum, module->lastMode);
            module->bit8dataNum++;
        }
        //(*dataNum)++;
        module->status = 0;
        module->upperNANOSEC = NANOSEC;
        module->lowerNANOSEC = NANOSEC;
    }

    //printf("ACQMode = %u, LastMode = %u, Mode = %u, ModuleNum = %u, QuaboNum = %u, UTC = %u, NANOSEC = %u, PKTNUM = %u\n", acqmode, module->lastMode, mode, moduleNum, quaboNum, UTC, NANOSEC, PKTNUM);
    storePktData((uint8_t *)module->data, data_ptr, mode, quaboIndex);
    module->lastMode = mode;
    module->PKTNUM[quaboIndex] = PKTNUM;
    //module->UTC[quaboIndex] = UTC;
    module->tv_sec[quaboIndex] = tv_sec;
    module->tv_usec[quaboIndex] = tv_usec;
    module->NANOSEC[quaboIndex] = NANOSEC;

    module->status = module->status | currentStatus;
}

/**
 * Close the HDF5 file.
 */
void closeFile(fileIDs_t* file){
    H5Gclose(file->bit16IMGData);
    H5Gclose(file->bit8IMGData);
    H5Gclose(file->PHData);
    H5Gclose(file->ShortTransient);
    H5Gclose(file->bit16HCData);
    H5Gclose(file->bit8HCData);
    H5Gclose(file->DynamicMeta);
    H5Gclose(file->StaticMeta);
    H5Fclose(file->file);
    free(file);
}

/**
 * Close all of the modules that were initalized.
 */
void closeModules(modulePairData_t* head){
    modulePairData_t* currentmodule;
    currentmodule = head;
    while (head != NULL){
        moduleFillZeros(currentmodule, currentmodule->status);
        if(currentmodule->lastMode == 16){
            writeDataBlock(currentmodule->ID16bit, currentmodule, currentmodule->bit16dataNum, 16);
        } else if (currentmodule->lastMode == 8){
            writeDataBlock(currentmodule->ID8bit, currentmodule, currentmodule->bit8dataNum, 8);
        }
        H5Gclose(head->ID16bit);
        H5Gclose(head->ID8bit);
        H5Gclose(head->dynamicMeta);
        head = head->next_moduleID;
        free(currentmodule);
        printf("Flushed and Closed Module %u and %u\n", currentmodule->mod1Name, currentmodule->mod2Name);
        currentmodule = head;
    }
}

/**
 * Initalize the HDF5 file given a name and time.
 */
fileIDs_t* HDF5file_init(char* fileName, char* currTime){
    fileIDs_t* new_file;
    FILE *modConfig_file;
    hid_t datatype, dataspace;
    hsize_t dimsf[2];
    char fbuf[100];
    char cbuf;
    unsigned int mod1Name;
    unsigned int mod2Name;
    
    
    new_file = createNewFile(fileName, currTime);
    
    createDMetaResources(new_file->DynamicMeta);

    moduleListBegin = modulePairData_t_new();
    moduleListEnd = moduleListBegin;

    modConfig_file = fopen(CONFIGFILE, "r");
    if (modConfig_file == NULL) {
        perror("Error Opening File\n");
        exit(1);
    }

    dimsf[0] = PKTPERPAIR;
    dimsf[1] = PKTSIZE;
    dataspace = H5Screate_simple(RANK, dimsf, NULL);
    datatype = H5Tcopy(H5T_STD_U16LE);

    cbuf = getc(modConfig_file);
    char moduleName[50];

    while(cbuf != EOF){
        ungetc(cbuf, modConfig_file);
        if (cbuf != '#'){
            if (fscanf(modConfig_file, "%u %u\n", &mod1Name, &mod2Name) == 2){
                if (moduleInd[mod1Name] == NULL && moduleInd[mod2Name] == NULL){

                    sprintf(moduleName, MODULEPAIR_FORMAT, mod1Name, mod2Name);

                    moduleInd[mod1Name] = moduleInd[mod2Name] = moduleListEnd->next_moduleID 
                                            = modulePairData_t_new(createModPair(new_file->bit16IMGData, mod1Name, mod2Name), 
                                                                    createModPair(new_file->bit8IMGData, mod1Name, mod2Name),
                                                                    createModPair(new_file->DynamicMeta, mod1Name, mod2Name), 
                                                                    mod1Name, mod2Name);
                    
                    moduleListEnd = moduleListEnd->next_moduleID;
                    
                    createQuaboTables(moduleListEnd->dynamicMeta, moduleListEnd);

                    printf("Created Module Pair: %u.%u-%u and %u.%u-%u\n", 
                    (unsigned int) (mod1Name << 2)/0x100, (mod1Name << 2) % 0x100, ((mod1Name << 2) % 0x100) + 3,
                    (mod2Name << 2)/0x100, (mod2Name << 2) % 0x100, ((mod2Name << 2) % 0x100) + 3);
                }
            }
        } else {
            if (fgets(fbuf, 100, modConfig_file) == NULL){
                break;
            }
        }
        cbuf = getc(modConfig_file);
    }

    if (fclose(modConfig_file) == EOF){
        printf("Warning: Unable to close module configuration file.\n");
    }
    if (H5Sclose(dataspace) < 0){
        printf("Warning: Unable to close HDF5 file database.\n");
    }
    if (H5Tclose(datatype) < 0){
        printf("Warning: Unable to close HDF5 file datatype.\n");
    }
    return new_file;
}

/**
 * Initilzing the HDF5 file based on the current file_naming format.
 */
fileIDs_t* HDF5file_init(){
    time_t t = time(NULL);
    struct tm tm = *gmtime(&t);
    char currTime[100];
    char fileName[100];
    
    sprintf(fileName, "%04i/", (tm.tm_year + 1900));
    mkdir(fileName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    sprintf(fileName, "%04i/%04i%02i%02i/", (tm.tm_year + 1900), tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
    mkdir(fileName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    //sprintf(fileName+strlen(fileName), "%04i%02i%02i", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
    sprintf(currTime, TIME_FORMAT,tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    sprintf(fileName+strlen(fileName), H5FILE_NAME_FORMAT, OBSERVATORY, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

    return HDF5file_init(fileName, currTime);
}

/**
 * Close and flush out all of the file resources
 */
void closeFileResources(){
    printf("-----Start Flushing and Closing all File Resources----\n");
    closeModules(moduleListBegin->next_moduleID);
    free(moduleListBegin);
    printf("--------------Closing HDF5 file--------------\n");
    closeFile(file);
}

/**
 * Close and flush out all of the resources allocated.
 */
void closeAllResources(){
    //printf("===FLUSHING ALL RESOURCES IN BUFFER===\n");
    //flushModules(moduleListBegin->next_moduleID);
    printf("\n===CLOSING ALL RESOURCES===\n");
    closeFileResources();
    //fclose(HSD_file);
    printf("\n-----------Closing Redis Connection-----------\n\n");
    redisFree(redisServer);
    //printf("Caught signal %d, coming out...\n", signum);
}

/**
 * Reinitlize the rfile resources by first closing and flushing all resources then redefining them.
 */
void reinitFileResources(){
    time_t t = time(NULL);
    struct tm tm = *gmtime(&t);
    char currTime[100];
    char fileName[100];

    sprintf(fileName, "%04i/", (tm.tm_year + 1900));
    mkdir(fileName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    sprintf(fileName, "%04i/%04i%02i%02i/", (tm.tm_year + 1900), tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
    mkdir(fileName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    sprintf(currTime, TIME_FORMAT,tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    sprintf(fileName+strlen(fileName), H5FILE_NAME_FORMAT, OBSERVATORY, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

    if (access(fileName, F_OK) != -1){
        return;
    }
    printf("\n===CLOSING FILE RESOURCES===\n");
    closeFileResources();
    printf("\n===INITIALING FILE RESROUCES===\n");
    file = HDF5file_init(fileName, currTime);
    printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n");
}

//Signal handeler to allow for hashpipe to exit gracfully and also to allow for creating of new files by command.
static int QUITSIG;

void QUIThandler(int signum) {
    QUITSIG = 1;
}

static void *run(hashpipe_thread_args_t * args){

    signal(SIGQUIT, QUIThandler);

    QUITSIG = 0;

    /*Initialization of HASHPIPE Values*/
    // Local aliases to shorten access to args fields
    // Our input buffer happens to be a demo1_ouput_databuf
    HSD_output_databuf_t *db = (HSD_output_databuf_t *)args->ibuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    int rv;
    int block_idx = 0;
    uint64_t mcnt=0;

    //Output elements
    char *block_ptr;
    //char *textblock = (char *)malloc((BLOCKSIZE*sizeof(char)*3 + N_PKT_PER_BLOCK));
    int packetNum = 0;
    uint16_t moduleNum;
    uint8_t quaboNum;
    char acqmode;
    uint16_t packet_NUM;
    uint32_t packet_UTC;
    uint32_t packet_NANOSEC;

    int maxSizeInput = 0;

    hgeti4(st.buf, "MAXFILESIZE", &maxSizeInput);
    maxFileSize = maxSizeInput*8E5; 

    
    /*Initialization of Redis Server Values*/
    printf("------------------SETTING UP REDIS ------------------\n");
    redisServer = redisConnect("127.0.0.1", 6379);
    if (redisServer != NULL && redisServer->err){
        printf("Error: %s\n", redisServer->errstr);
        exit(1);
    } else {
        printf("Connect to Redis\n");
    }
    redisReply *keysReply;
    redisReply *reply;
    char command[50];
    // Uncomment following lines for redis servers with password
    // reply = redisCommand(redisServer, "AUTH password");
    // freeReplyObject(reply);


    /* Initialization of HDF5 Values*/
    printf("-------------------SETTING UP HDF5 ------------------\n");
    modulePairData_t* currentModule;
    file = HDF5file_init();
    getStaticRedisData(redisServer, file->StaticMeta);
    //get_storeGPSSupp(redisServer, file->StaticMeta);
    
    getDynamicRedisData(redisServer, moduleListBegin->next_moduleID, file->DynamicMeta);
    HKPackets_t* HK = (HKPackets_t *)malloc(sizeof(struct HKPackets));
    /*HK->BOARDLOC[0] = 504;
    HK->BOARDLOC[1] = 503;
    printf("Testing on mod %u\n", moduleListBegin->next_moduleID->mod1Name);
    printf("Testing on mod %u\n", moduleListBegin->next_moduleID->mod2Name);
    fetchHKdata(HK, redisServer);
    printf("Module ID is %li and %li", moduleListEnd->ID16bit, moduleListEnd->ID8bit);
    createDataBlock(moduleListEnd, HK);
    createDataBlock(moduleListEnd, HK);*/
    //storeHKdata(moduleListBegin->next_moduleID->ID16bit, HK);

    printf("-----------Finished Setup of Output Thread-----------\n");
    printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n\n");

    /* Main loop */
    while(run_threads()){

        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "OUTBLKIN", block_idx);
	    hputi8(st.buf, "OUTMCNT",mcnt);
        hputs(st.buf, status_key, "waiting");
        hashpipe_status_unlock_safe(&st);

        //Wait for the output buffer to be free
        while ((rv=HSD_output_databuf_wait_filled(db, block_idx))
                != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked");
                hashpipe_status_unlock_safe(&st);
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
                pthread_exit(NULL);
                break;
            }
        }

        // Mark the buffer as processing
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "processing");
        hashpipe_status_unlock_safe(&st);

        //TODO check mcnt
        //Read the packet
        block_ptr=db->block[block_idx].stream_block;

        getDynamicRedisData(redisServer, moduleListBegin->next_moduleID, file->DynamicMeta);
        for(int i = 0; i < db->block[block_idx].header.stream_block_size; i++){
            acqmode = db->block[block_idx].header.acqmode[i];
            packet_NUM = db->block[block_idx].header.pktNum[i];
            moduleNum = db->block[block_idx].header.modNum[i];
            quaboNum = db->block[block_idx].header.quaNum[i];
            packet_UTC = db->block[block_idx].header.pktUTC[i]; 
            packet_NANOSEC = db->block[block_idx].header.pktNSEC[i];


            if (moduleInd[moduleNum] == NULL){

                printf("Detected New Module not in Config File: %u.%u\n", (unsigned int) (moduleNum << 2)/0x100, (moduleNum << 2) % 0x100);

                moduleInd[moduleNum] = moduleListEnd->next_moduleID 
                                            = modulePairData_t_new(createModPair(file->bit16IMGData, moduleNum, 0), 
                                                                    createModPair(file->bit8IMGData, moduleNum, 0),
                                                                    createModPair(file->DynamicMeta, moduleNum, 0), 
                                                                    moduleNum);
                moduleListEnd = moduleListEnd->next_moduleID;
                currentModule = moduleListEnd;

            } else {
                currentModule = moduleInd[moduleNum];
            }

            storeData(currentModule, acqmode, moduleNum, quaboNum, packet_NUM, packet_UTC, packet_NANOSEC,
                        db->block[block_idx].header.tv_sec[i], db->block[block_idx].header.tv_usec[i], block_ptr + (i*PKTDATASIZE));
        }

        /* Term conditions */

        if (db->block[block_idx].header.INTSIG){
            closeAllResources();
            exit(0);
        }

        if (QUITSIG || fileSize > maxFileSize) {
            reinitFileResources();
            getStaticRedisData(redisServer, file->StaticMeta);
            //get_storeGPSSupp(redisServer, file->StaticMeta);
            fileSize = 0;
            QUITSIG = 0;
        }

        HSD_output_databuf_set_free(db,block_idx);
	    block_idx = (block_idx + 1) % db->header.n_block;
	    mcnt++;

        //Will exit if thread has been cancelled
        pthread_testcancel();
    }

    printf("===CLOSING ALL RESOURCES===\n");
    closeFile(file);
    //fclose(HSD_file);
    redisFree(redisServer);
    return THREAD_OK;
}

/**
 * Sets the functions and buffers for this thread
 */
static hashpipe_thread_desc_t HSD_output_thread = {
    name: "HSD_output_thread",
    skey: "OUTSTAT",
    init: NULL,
    run: run,
    ibuf_desc: {HSD_output_databuf_create},
    obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&HSD_output_thread);
}

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
#include "hashpipe.h"
#include "HSD_databuf.h"
#include "hiredis/hiredis.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#define H5FILE_NAME_FORMAT "PANOSETI_%s_%04i_%02i_%02i_%02i-%02i-%02i.h5"
#define TIME_FORMAT "%04i-%02i-%02iT%02i:%02i:%02i UTC"
#define OBSERVATORY "LICK"
#define RANK 2
#define CONFIGFILE "./modulePair.config"
#define FRAME_FORMAT "Frame%05i"
#define DATA_FORMAT "DATA%09i"
#define QUABO_FORMAT "QUABO%05i_%01i"
#define HK_TABLENAME_FORAMT "HK_Module%05i_Quabo%01i"
#define HK_TABLETITLE_FORMAT "HouseKeeping Data for Module%05i_Quabo%01i"
#define MODULEPAIR_FORMAT "ModulePair_%05u_%05u"

#define QUABOPERMODULE 4
#define PKTPERPAIR QUABOPERMODULE*2
#define SCIDATASIZE 256
#define HKDATASIZE 464
#define DATABLOCKSIZE SCIDATASIZE*PKTPERPAIR+64+16
#define HKFIELDS 27
#define NANOSECTHRESHOLD 20

#define STRBUFFSIZE 50

static hsize_t storageDim[RANK] = {PKTPERPAIR,SCIDATASIZE};

static hid_t storageSpace = H5Screate_simple(RANK, storageDim, NULL);

static hid_t storageType = H5Tcopy(H5T_STD_U16LE);

static long long fileSize = 0;

static long long maxFileSize = 0; //IN UNITS OF APPROX 2 BYTES OR 16 bits

typedef struct fileIDs {
    hid_t       file;         /* file and dataset handles */
    hid_t       bit16IMGData, bit8IMGData, PHData, ShortTransient, bit16HCData, bit8HCData, DynamicMeta;
} fileIDs_t;

typedef struct HKPackets {
    char SYSTIME[STRBUFFSIZE];
    uint16_t BOARDLOC;
    int16_t HVMON0, HVMON1, HVMON2, HVMON3;
    uint16_t HVIMON0, HVIMON1, HVIMON2, HVIMON3;
    int16_t RAWHVMON;
    uint16_t V12MON, V18MON, V33MON, V37MON;
    uint16_t I10MON, I18MON, I33MON;
    int16_t TEMP1;
    uint16_t TEMP2;
    uint16_t VCCINT, VCCAUX;
    uint16_t UID;
    uint8_t SHUTTER_STATUS, LIGHT_STATUS;
    uint16_t FWID0, FWID1;
} HKPackets_t;

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

hid_t get_H5T_string_type(){
    hid_t string_type;

    string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size( string_type, STRBUFFSIZE );
    return string_type;
}

const hid_t HK_field_types[HKFIELDS] = { get_H5T_string_type(), H5T_STD_U16LE,                  // SYSTIME, BOARDLOC
                                H5T_STD_I16LE, H5T_STD_I16LE, H5T_STD_I16LE, H5T_STD_I16LE,     // HVMON0-3
                                H5T_STD_U16LE, H5T_STD_U16LE, H5T_STD_U16LE, H5T_STD_U16LE,     // HVIMON0-3
                                H5T_STD_I16LE,                                                  // RAWHVMON
                                H5T_STD_U16LE, H5T_STD_U16LE, H5T_STD_U16LE, H5T_STD_U16LE,     // V12MON, V18MON, V33MON, V37MON           
                                H5T_STD_U16LE, H5T_STD_U16LE, H5T_STD_U16LE,                    // I10MON, I18MON, I33MON        
                                H5T_STD_I16LE, H5T_STD_U16LE,                                   // TEMP1, TEMP2                        
                                H5T_STD_U16LE, H5T_STD_U16LE,                                   // VCCINT, VCCAUX              
                                H5T_STD_U16LE,                                                  // UID
                                H5T_STD_I8LE,H5T_STD_I8LE,                                      // SHUTTER and LIGHT_SENSOR STATUS
                                H5T_STD_U16LE,H5T_STD_U16LE                                     // FWID0 and FWID1
};

typedef struct moduleIDs {
    hid_t ID16bit;
    hid_t ID8bit;
    hid_t dynamicMeta;
    uint8_t status;   // Determine the which part of the data is filled 0:neither filled 1:First rank filled 2: Second rank filled
    int lastMode;
    int mod1Name;
    int mod2Name;
    uint16_t data[PKTPERPAIR][SCIDATASIZE];
    uint16_t PKTNUM[PKTPERPAIR];
    uint32_t UTC[PKTPERPAIR];
    uint32_t NANOSEC[PKTPERPAIR];
    uint32_t upperNANOSEC;
    uint32_t lowerNANOSEC;
    int bit16dataNum;
    int bit8dataNum;
    moduleIDs* next_moduleID;
} moduleIDs_t;


moduleIDs_t* moduleIDs_t_new(hid_t ID16, hid_t ID8, hid_t dynamicMD, unsigned int mod1, unsigned int mod2){
    moduleIDs_t* value = (moduleIDs_t*) malloc(sizeof(struct moduleIDs));
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

moduleIDs_t* moduleIDs_t_new(){
    moduleIDs_t_new(0,0,0,-1,-1);
}

moduleIDs_t* moduleIDs_t_new(hid_t ID16, hid_t ID8, hid_t dynamicMD, unsigned mod1){
    moduleIDs_t_new(ID16, ID8, dynamicMD, mod1, -1);
}

moduleIDs_t* get_moduleID(moduleIDs_t* list, unsigned int ind){
    if(list != NULL && ind > 0)
        return get_moduleID(list->next_moduleID, ind-1);
    return list;
}

void moduleFillZeros(moduleIDs_t* module, uint8_t status){//uint16_t data[PKTPERPAIR][SCIDATASIZE], uint8_t status){
    for(int i = 0; i < PKTPERPAIR; i++){
        if(!((status >> i) & 0x01)){
            memset(module->data[i], 0, SCIDATASIZE*sizeof(uint16_t));
            module->PKTNUM[i] = 0;
            module->UTC[i] = 0;
            module->NANOSEC[i] = 0;
        }
    }
}

static char hex_to_char(char in){
    switch (in) {
        case 0x00: return '0';
        case 0x01: return '1';
        case 0x02: return '2';
        case 0x03: return '3';
        case 0x04: return '4';
        case 0x05: return '5';
        case 0x06: return '6';
        case 0x07: return '7';
        case 0x08: return '8';
        case 0x09: return '9';
        case 0x0a: return 'a';
        case 0x0b: return 'b';
        case 0x0c: return 'c';
        case 0x0d: return 'd';
        case 0x0e: return 'e';
        case 0x0f: return 'f';
        default: return '=';
    }
}

static void data_to_text(char *data, char *text){
    int textInd;
    for(int i = 0; i < BLOCKSIZE; i++){
        textInd = i*3;
        text[textInd] = hex_to_char(((data[i] >> 4) & 0x0f));
        text[textInd + 1] = hex_to_char(data[i] & 0x0f);
        if (i % PKTSIZE == PKTSIZE-1) {
            text[textInd + 2] = '\n';
        } else {
            text[textInd + 2] = ' ';
        }
    }
}

void createStrAttribute(hid_t group, const char* name, char* data) {
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;

    dataspace = H5Screate(H5S_SCALAR);

    datatype = H5Tcopy(H5T_C_S1);
    H5Tset_size(datatype, strlen(data));
    H5Tset_strpad(datatype, H5T_STR_NULLTERM);
    H5Tset_cset(datatype, H5T_CSET_UTF8);

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);

    H5Awrite(attribute, datatype, data);
    // Add size to fileSize
    fileSize += STRBUFFSIZE;

    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Aclose(attribute);

}

void createStrAttribute2(hid_t group, const char* name, hsize_t* dimsf, char data[PKTPERPAIR][STRBUFFSIZE]) {
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;

    dataspace = H5Screate_simple(sizeof(dimsf)/sizeof(dimsf[0]), dimsf, NULL);

    datatype = H5Tcopy(H5T_C_S1);
    H5Tset_size(datatype, STRBUFFSIZE);
    H5Tset_strpad(datatype, H5T_STR_NULLTERM);
    H5Tset_cset(datatype, H5T_CSET_UTF8);

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, datatype, data[0]);
    fileSize += STRBUFFSIZE;

    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Aclose(attribute);
}

void createNumAttribute(hid_t group, const char* name, hid_t dtype, unsigned long long data) {
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;
    unsigned long long attr_data[1];
    attr_data[0] = data;

    
    dataspace = H5Screate(H5S_SCALAR);

    datatype = H5Tcopy(dtype);

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);

    H5Awrite(attribute, dtype, attr_data);
    fileSize += 16;
    
    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Aclose(attribute);
}

void createNumAttribute2(hid_t group, const char* name, hid_t dtype, hsize_t* dimsf, void* data) {
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;

    dataspace = H5Screate_simple(sizeof(dimsf)/sizeof(dimsf[0]), dimsf, NULL);
    datatype = H5Tcopy(dtype);

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, dtype, data);
    fileSize += 32;

    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Aclose(attribute);
}

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

    //createNumAttribute(modulePair, "Module1", H5T_STD_U64LE , mod1Name);
    //createNumAttribute(modulePair, "Module2", H5T_STD_U64LE , mod2Name);

    createNumAttribute2(modulePair, "ModuleNum", H5T_STD_U64LE, dimsf, modNames);

    return modulePair;//H5Gclose(modulePair);

}

hid_t createMod(hid_t group, unsigned int mod1Name){
    hid_t   modulePair;
    char    modName[STRBUFFSIZE];

    sprintf(modName, "./ModulePair_%05u", mod1Name);

    modulePair = H5Gcreate(group, modName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    createNumAttribute(modulePair, "ModuleNum", H5T_STD_U64LE , mod1Name);

    return modulePair;
}

void createQuaboTables(hid_t group, moduleIDs_t* module){

    /*HKPackets_t HK_data[1] = {{ "null", module->mod1Name,
                                -1, -1, -1, -1,         // HVMON (0 to -80V)
                                0, 0, 0, 0,             // HVIMON ((65535-HVIMON) * 38.1nA) (0 to 2.5mA)
                                -1,                     // RAWHVMON (0 to -80V)
                                0,                      // V12MON (19.07uV/LSB) (1.2V supply)
                                0,                      // V18MON (19.07uV/LSB) (1.8V supply)
                                0,                      // V33MON (38.10uV/LSB) (3.3V supply)
                                0,                      // V37MON (38.10uV/LSB) (3.7V supply)
                                0,                      // I10MON (182uA/LSB) (1.0V supply)
                                0,                      // I18MON (37.8uA/LSB) (1.8V supply)
                                0,                      // I33MON (37.8uA/LSB) (3.3V supply)
                                -1,                     // TEMP1 (0.0625*N)
                                0,                      // TEMP2 (N/130.04-273.15)
                                0,                      // VCCINT (N*3/65536)
                                0,                      // VCCAUX (N*3/65536)
                                0,                      // UID
                                0,0,                    // SHUTTER and LIGHT_SENSOR STATUS
                                0,0                     // FWID0 and FWID1

    }};*/

    HKPackets_t HK_data;
    char tableName[50];
    char tableTitle[50];
    for (int i = 0; i < QUABOPERMODULE; i++) {
        sprintf(tableName, HK_TABLENAME_FORAMT, module->mod1Name, i);
        sprintf(tableTitle, HK_TABLETITLE_FORMAT, module->mod1Name, i);

        H5TBmake_table(tableTitle, group, tableName, HKFIELDS, 0,
                            HK_dst_size, HK_field_names, HK_dst_offset, HK_field_types,
                            100, NULL, 0, &HK_data);
    }

    for (int i = 0; i < QUABOPERMODULE; i++) {
        sprintf(tableName, HK_TABLENAME_FORAMT, module->mod2Name, i);
        sprintf(tableTitle, HK_TABLETITLE_FORMAT, module->mod2Name, i);

        H5TBmake_table(tableTitle, group, tableName, HKFIELDS, 0,
                            HK_dst_size, HK_field_names, HK_dst_offset, HK_field_types,
                            100, NULL, 0, &HK_data);
    }
}

fileIDs_t* createNewFile(char* fileName, char* currTime){
    fileIDs_t* newfile = (fileIDs_t*) malloc(sizeof(struct fileIDs));

    newfile->file = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    createStrAttribute(newfile->file, "dateCreated", currTime);
    newfile->bit16IMGData = H5Gcreate(newfile->file, "/bit16IMGData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->bit8IMGData = H5Gcreate(newfile->file, "/bit8IMGData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->PHData = H5Gcreate(newfile->file, "/PHData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->ShortTransient = H5Gcreate(newfile->file, "/ShortTransient", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->bit16HCData =  H5Gcreate(newfile->file, "/bit16HCData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile->bit8HCData = H5Gcreate(newfile->file, "/bit8HCData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    newfile->DynamicMeta = H5Gcreate(newfile->file, "/DynamicMeta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    return newfile;
}

void fetchHKdata(HKPackets_t* HK, uint16_t BOARDLOC, redisContext* redisServer) {
    redisReply *reply;
    char command[50];
    sprintf(command, "HGET %u %s", BOARDLOC, "SYSTIME");
    reply = (redisReply *)redisCommand(redisServer, command);
    strcpy(HK->SYSTIME, reply->str);
    freeReplyObject(reply);

    sprintf(command, "HGET %u %s", BOARDLOC, "BOARDLOC");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->BOARDLOC = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "HVMON0");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->HVMON0 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "HVMON1");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->HVMON1 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "HVMON2");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->HVMON2 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "HVMON3");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->HVMON3 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);

    sprintf(command, "HGET %u %s", BOARDLOC, "HVIMON0");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->HVIMON0 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "HVIMON1");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->HVIMON1 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "HVIMON2");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->HVIMON2 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "HVIMON3");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->HVIMON3 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);

    sprintf(command, "HGET %u %s", BOARDLOC, "RAWHVMON");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->RAWHVMON = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);

    sprintf(command, "HGET %u %s", BOARDLOC, "V12MON");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->V12MON = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "V18MON");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->V18MON = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "V33MON");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->V33MON = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "V37MON");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->V37MON = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);

    sprintf(command, "HGET %u %s", BOARDLOC, "I10MON");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->I10MON = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "I18MON");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->I18MON = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "I33MON");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->I33MON = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);

    sprintf(command, "HGET %u %s", BOARDLOC, "TEMP1");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->TEMP1 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "TEMP2");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->TEMP2 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);

    sprintf(command, "HGET %u %s", BOARDLOC, "VCCINT");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->VCCINT = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "VCCAUX");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->VCCAUX = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);

    sprintf(command, "HGET %u %s", BOARDLOC, "UID");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->UID = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);

    sprintf(command, "HGET %u %s", BOARDLOC, "SHUTTER_STATUS");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->SHUTTER_STATUS = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "LIGHT_SENSOR_STATUS");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->LIGHT_STATUS = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);

    sprintf(command, "HGET %u %s", BOARDLOC, "FWID0");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->FWID0 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
    sprintf(command, "HGET %u %s", BOARDLOC, "FWID1");
    reply = (redisReply *)redisCommand(redisServer, command);
    HK->FWID1 = strtoll(reply->str, NULL, 10);
    freeReplyObject(reply);
}

void check_storeHK(redisContext* redisServer, moduleIDs_t* modHead){
    HKPackets_t* HKdata = (HKPackets_t *)malloc(sizeof(HKPackets));
    moduleIDs_t* currentMod;
    redisReply* reply;
    uint16_t BOARDLOC;
    char tableName[50];
    char command[50];

    currentMod = modHead;
    
    while(currentMod != NULL){

        //Updating all the Quabos from Module 1
        BOARDLOC = (currentMod->mod1Name << 2) & 0xfffc;
        
        for(int i = 0; i < 4; i++){
            sprintf(command, "HGET UPDATED %u", BOARDLOC);
            reply = (redisReply *)redisCommand(redisServer, command);

            if (strtol(reply->str, NULL, 10)){
                freeReplyObject(reply);

                fetchHKdata(HKdata, BOARDLOC, redisServer);
                sprintf(tableName, HK_TABLENAME_FORAMT, currentMod->mod1Name, i);
                H5TBappend_records(currentMod->dynamicMeta, tableName, 1, HK_dst_size, HK_dst_offset, HK_dst_sizes, HKdata);
                fileSize += HKDATASIZE;

                sprintf(command, "HSET UPDATED %u 0", BOARDLOC);
                reply = (redisReply *)redisCommand(redisServer, command);
            }

            freeReplyObject(reply);
            BOARDLOC++;
        }

        if(currentMod->mod2Name != -1){
            //Updating all the Quabos from Module 2
            BOARDLOC = (currentMod->mod2Name << 2) & 0xfffc;

            for(int i = 0; i < 4; i++){
                sprintf(command, "HGET UPDATED %u", BOARDLOC);
                reply = (redisReply *)redisCommand(redisServer, command);

                if (strtol(reply->str, NULL, 10)){
                    freeReplyObject(reply);

                    fetchHKdata(HKdata, BOARDLOC, redisServer);
                    sprintf(tableName, HK_TABLENAME_FORAMT, currentMod->mod2Name, i);
                    H5TBappend_records(currentMod->dynamicMeta, tableName, 1, HK_dst_size, HK_dst_offset, HK_dst_sizes, HKdata);

                    sprintf(command, "HSET UPDATED %u 0", BOARDLOC);
                    reply = (redisReply *)redisCommand(redisServer, command);
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

void writeDataBlock(hid_t frame, moduleIDs_t* module, int index){
    hid_t dataset;
    char name[50];
    hsize_t dimsf[1];
    dimsf[0] = PKTPERPAIR;

    sprintf(name, DATA_FORMAT, index);
    dataset = H5Dcreate2(frame, name, storageType, storageSpace,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    H5Dwrite(dataset, H5T_STD_U16LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, module->data);

    createNumAttribute2(dataset, "PKTNUM", H5T_STD_U16LE, dimsf, module->PKTNUM);
    createNumAttribute2(dataset, "UTC", H5T_STD_U32LE, dimsf, module->UTC);
    createNumAttribute2(dataset, "NANOSEC", H5T_STD_U32LE, dimsf, module->NANOSEC);

    fileSize += DATABLOCKSIZE;

    H5Dclose(dataset);
}

void storePktData(uint16_t* moduleData, char* data_ptr, int mode){

    if (mode == 16){
        for(int i = 0; i < SCIDATASIZE; i++){
            moduleData[i] = (data_ptr[i*2 + 1] << 8) & 0xff00 | (data_ptr[i*2] & 0x00ff);
        }
    } else if(mode == 8){
        for(int i = 0; i < SCIDATASIZE; i++){
            moduleData[i] = data_ptr[i];
        }
    } else {
        return;
    }
}

void storeData(moduleIDs_t* module, char acqmode, uint16_t moduleNum, uint8_t quaboNum, uint16_t PKTNUM, uint32_t UTC, uint32_t NANOSEC, char* data_ptr){
    //uint16_t* moduleData;
    int* dataNum;
    hid_t group;
    int mode;
    int quaboIndex;
    uint8_t currentStatus = (0x01 << quaboNum);
    //printf("Module %u, Quabo %u\n", moduleNum, quaboNum);

    if(acqmode == 0x2 || acqmode == 0x3){
        group = module->ID16bit;
        dataNum = &(module->bit16dataNum);
        mode = 16;
    } else if (acqmode == 0x6 || acqmode == 0x7){
        group = module->ID8bit;
        dataNum = &(module->bit8dataNum);
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
        printf("\n");
        moduleFillZeros(module, module->status);
        writeDataBlock(group, module, *dataNum);
        (*dataNum)++;
        module->status = 0;
        module->upperNANOSEC = NANOSEC;
        module->lowerNANOSEC = NANOSEC;
    }

    printf("ModuleNum = %u, QuaboNum = %u, UTC = %u, NANOSEC = %u uNANOSEC = %u lNANOSEC = %u\n", moduleNum, quaboNum, UTC, NANOSEC, module->upperNANOSEC, module->lowerNANOSEC);
    storePktData(module->data[quaboIndex], data_ptr, mode);
    module->lastMode = mode;
    module->PKTNUM[quaboIndex] = PKTNUM;
    module->UTC[quaboIndex] = UTC;
    module->NANOSEC[quaboIndex] = NANOSEC;

    module->status = module->status | currentStatus;
}

moduleIDs_t* get_module_info(moduleIDs_t* list, unsigned int ind){
    if(list != NULL && ind > 0)
        return get_module_info(list->next_moduleID, ind-1);
    return list;
}

void closeFile(fileIDs_t* file){
    H5Fclose(file->file);
    H5Gclose(file->bit16IMGData);
    H5Gclose(file->bit8IMGData);
    H5Gclose(file->PHData);
    H5Gclose(file->ShortTransient);
    free(file);
}

void closeModules(moduleIDs_t* head){
    moduleIDs_t* currentmodule;
    currentmodule = head;
    while (head != NULL){
        moduleFillZeros(currentmodule, currentmodule->status);
        if(currentmodule->lastMode == 16){
            writeDataBlock(currentmodule->ID16bit, currentmodule, currentmodule->bit16dataNum);
        } else if (currentmodule->lastMode == 8){
            writeDataBlock(currentmodule->ID8bit, currentmodule, currentmodule->bit8dataNum);
        }
        H5Gclose(head->ID16bit);
        H5Gclose(head->ID8bit);
        head = head->next_moduleID;
        free(currentmodule);
        printf("Flushed and Closed Module %u and %u\n", currentmodule->mod1Name, currentmodule->mod2Name);
        currentmodule = head;
    }
}



static moduleIDs_t* moduleListBegin = moduleIDs_t_new();
static moduleIDs_t* moduleListEnd = moduleListBegin;
static unsigned int moduleListSize = 1;
static unsigned int moduleInd[0xffff];
static fileIDs_t* file;
static redisContext *redisServer;

    
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

    moduleListBegin = moduleIDs_t_new();
    moduleListEnd = moduleListBegin;
    moduleListSize = 1;
    memset(moduleInd, -1, sizeof(moduleInd));

    modConfig_file = fopen(CONFIGFILE, "r");
    if (modConfig_file == NULL) {
        perror("Error Opening File\n");
        exit(0);
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
                if (moduleInd[mod1Name] == -1 && moduleInd[mod2Name] == -1){
                    moduleInd[mod1Name] = moduleInd[mod2Name] = moduleListSize;
                    moduleListSize++;

                    printf("Created Module Pair: %u.%u-%u and %u.%u-%u\n", 
                    (unsigned int) (mod1Name << 2)/0x100, (mod1Name << 2) % 0x100, ((mod1Name << 2) % 0x100) + 3,
                    (mod2Name << 2)/0x100, (mod2Name << 2) % 0x100, ((mod2Name << 2) % 0x100) + 3);

                    sprintf(moduleName, MODULEPAIR_FORMAT, mod1Name, mod2Name);

                    moduleListEnd->next_moduleID = moduleIDs_t_new(createModPair(new_file->bit16IMGData, mod1Name, mod2Name), 
                                                                    createModPair(new_file->bit8IMGData, mod1Name, mod2Name),
                                                                    createModPair(new_file->DynamicMeta, mod1Name, mod2Name), 
                                                                    mod1Name, mod2Name);
                    
                    moduleListEnd = moduleListEnd->next_moduleID;
                    
                    createQuaboTables(moduleListEnd->dynamicMeta, moduleListEnd);
                }
            }
        } else {
            if (fgets(fbuf, 100, modConfig_file) == NULL){
                break;
            }
        }
        cbuf = getc(modConfig_file);
    }
    fclose(modConfig_file);
    H5Sclose(dataspace);
    H5Tclose(datatype);
    return new_file;
}

fileIDs_t* HDF5file_init(){
    time_t t = time(NULL);
    struct tm tm = *gmtime(&t);
    char currTime[100];
    char fileName[100];

    sprintf(currTime, TIME_FORMAT,tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    sprintf(fileName, H5FILE_NAME_FORMAT, OBSERVATORY, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

    return HDF5file_init(fileName, currTime);
}

void closeFileResources(){
    printf("--------------Closing HDF5 file--------------\n");
    closeFile(file);
    printf("-----Start Flushing and Closing all Files----\n");
    closeModules(moduleListBegin->next_moduleID);
    free(moduleListBegin);
}

void closeAllResources(){
    //printf("===FLUSHING ALL RESOURCES IN BUFFER===\n");
    //flushModules(moduleListBegin->next_moduleID);
    printf("\n===CLOSING ALL RESOURCES===\n");
    closeFileResources();
    //fclose(HSD_file);
    printf("\n-----------Closing Redis Connection-----------\n\n");
    redisFree(redisServer);
    //printf("Caught signal %d, coming out...\n", signum);
    exit(1);
}

void reinitFileResources(){
    time_t t = time(NULL);
    struct tm tm = *gmtime(&t);
    char currTime[100];
    char fileName[100];

    sprintf(currTime, TIME_FORMAT,tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    sprintf(fileName, H5FILE_NAME_FORMAT, OBSERVATORY, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    
    if (access(fileName, F_OK) != -1){
        return;
    }
    printf("\n===CLOSING FILE RESOURCES===\n");
    closeFileResources();
    printf("\n===INITIALING FILE RESROUCES===\n");
    file = HDF5file_init(fileName, currTime);
}

static int INTSIG;
static int QUITSIG;

void INThandler(int signum) {
    INTSIG = 1;
}

void QUIThandler(int signum){
    QUITSIG = 1;
}

static void *run(hashpipe_thread_args_t * args){

    signal(SIGINT, INThandler);

    signal(SIGQUIT, QUIThandler);

    INTSIG = 0;
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
    char *textblock = (char *)malloc((BLOCKSIZE*sizeof(char)*3 + N_PKT_PER_BLOCK));
    int packetNum = 0;
    uint16_t moduleNum;
    uint8_t quaboNum;
    char acqmode;
    uint16_t packet_NUM;
    uint32_t packet_UTC;
    uint32_t packet_NANOSEC;

    //FILE * HSD_file;
    //HSD_file=fopen("./data.out", "w");

    int maxSizeInput = 0;

    hgeti4(st.buf, "MAXFILESIZE", &maxSizeInput);
    maxFileSize = maxSizeInput*5E5; 

    
    /*Initialization of Redis Server Values*/
    printf("------------------SETTING UP REDIS ------------------\n");
    redisServer = redisConnect("127.0.0.1", 6379);
    if (redisServer != NULL && redisServer->err){
        printf("Error: %s\n", redisServer->errstr);
        exit(0);
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
    moduleIDs_t* currentModule;
    file = HDF5file_init();
    
    check_storeHK(redisServer, moduleListBegin->next_moduleID);
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

    printf("-----------Finished Setup of Output Thread-----------\n\n");

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
        block_ptr=db->block[block_idx].result_block;

        #ifdef PRINT_TXT
            data_to_text(block_ptr, textblock);

            fprintf(HSD_file, "----------------------------\n");
            fprintf(HSD_file, "BLOCK %i\n", packetNum);
            packetNum++;
            fwrite(textblock, (BLOCKSIZE*sizeof(char)*3), 1, HSD_file);
            fprintf(HSD_file, "\n\n");
        #endif
        //fwrite(block_ptr, BLOCKSIZE*sizeof(char), 1, HSD_file);

        check_storeHK(redisServer, moduleListBegin->next_moduleID);
        for(int i = 0; i < N_PKT_PER_BLOCK; i++){
            acqmode = block_ptr[i*PKTSIZE];
            packet_NUM = ((block_ptr[i*PKTSIZE+3] << 8) & 0xff00) | (block_ptr[i*PKTSIZE+2] & 0x00ff);
            moduleNum = ((block_ptr[i*PKTSIZE+5] << 6) & 0x3fc0) | ((block_ptr[i*PKTSIZE+4] >> 2) & 0x003f);
            quaboNum = ((block_ptr[i*PKTSIZE+4]) & 0x03);
            packet_UTC = ((block_ptr[i*PKTSIZE+9] << 24) & 0xff000000) 
                            | ((block_ptr[i*PKTSIZE+8] << 16) & 0x00ff0000)
                            | ((block_ptr[i*PKTSIZE+7] << 8) & 0x0000ff00)
                            | ((block_ptr[i*PKTSIZE+6]) & 0x000000ff);
                             
            packet_NANOSEC = ((block_ptr[i*PKTSIZE+13] << 24) & 0xff000000) 
                            | ((block_ptr[i*PKTSIZE+12] << 16) & 0x00ff0000)
                            | ((block_ptr[i*PKTSIZE+11] << 8) & 0x0000ff00)
                            | ((block_ptr[i*PKTSIZE+10]) & 0x000000ff);


            if (moduleInd[moduleNum] == -1){
                moduleInd[moduleNum] = moduleListSize;
                moduleListSize++;

                printf("Detected New Module not in Config File: %u.%u\n", (unsigned int) (moduleNum << 2)/0x100, (moduleNum << 2) % 0x100);

                moduleListEnd->next_moduleID = moduleIDs_t_new(createModPair(file->bit16IMGData, moduleNum, 0), 
                                                                    createModPair(file->bit8IMGData, moduleNum, 0),
                                                                    createModPair(file->DynamicMeta, moduleNum, 0), 
                                                                    moduleNum);
                moduleListEnd = moduleListEnd->next_moduleID;
                currentModule = moduleListEnd;

            } else {
                currentModule = get_module_info(moduleListBegin, moduleInd[moduleNum]);
            }

            storeData(currentModule, acqmode, moduleNum, quaboNum, packet_NUM, packet_UTC, packet_NANOSEC, block_ptr + (i*PKTSIZE+16));
        }

        /* Term conditions */

        if (INTSIG){
            closeAllResources();
        }

        if (QUITSIG || fileSize > maxFileSize) {
            reinitFileResources();
            fileSize = 0;
            QUITSIG = false;
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

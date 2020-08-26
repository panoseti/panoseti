/*
 * demo1_output_thread.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "hashpipe.h"
#include "HSD_databuf.h"
#include "hiredis/hiredis.h"
#include "hdf5.h"

#define H5FILE_NAME_FORMAT "PANOSETI_%s_%04i_%02i_%02i_%02i-%02i-%02i.h5"
#define OBSERVATORY "LICK"
#define RANK 2
#define NUMPKT 2
#define CONFIGFILE "./modulePair.config"
#define FRAME_FORMAT "Frame%05i"
#define DATA_FORMAT "DATA%05i"
#define STRBUFFSIZE 50
#define SCIDATASIZE 256

static hsize_t storageDim[RANK] = {NUMPKT,SCIDATASIZE};

static hid_t storageSpace = H5Screate_simple(RANK, storageDim, NULL);

static hid_t storageType = H5Tcopy(H5T_STD_U16LE);

typedef struct fileIDs {
    hid_t       file;         /* file and dataset handles */
    hid_t       bit16IMGData, bit8IMGData, PHData, ShortTransient, bit16HCData, bit8HCData, DynamicMeta;
} fileIDs_t;

typedef struct HKPackets {
    char SYSTIME[NUMPKT][STRBUFFSIZE];
    uint64_t BOARDLOC[NUMPKT];
    int64_t HVMON0[NUMPKT], HVMON1[NUMPKT], HVMON2[NUMPKT], HVMON3[NUMPKT];
    uint64_t HVIMON0[NUMPKT], HVIMON1[NUMPKT], HVIMON2[NUMPKT], HVIMON3[NUMPKT];
    int64_t RAWHVMON[NUMPKT];
    uint64_t V12MON[NUMPKT], V18MON[NUMPKT], V33MON[NUMPKT], V37MON[NUMPKT];
    uint64_t I10MON[NUMPKT], I18MON[NUMPKT], I33MON[NUMPKT];
    int64_t TEMP1[NUMPKT];
    uint64_t TEMP2[NUMPKT];
    uint64_t VCCINT[NUMPKT], VCCAUX[NUMPKT];
    uint64_t UID[NUMPKT];
    bool SHUTTER_STATUS[NUMPKT], LIGHT_STATUS[NUMPKT];
    uint64_t FWID0[NUMPKT], FWID1[NUMPKT];
} HKPackets_t;

typedef struct moduleIDs {
    hid_t ID16bit;
    hid_t ID8bit;
    //HKPackets_t HKData;
    unsigned int status;                 // Determine the which part of the data is filled 0:neither filled 1:First rank filled 2: Second rank filled
    int mod1Name;
    int mod2Name;
    uint16_t data[NUMPKT][SCIDATASIZE];
    int frameNum;
    int dataNum;
    hid_t ID16bitframe;
    hid_t ID8bitframe;
    moduleIDs* next_moduleID;
} moduleIDs_t;

moduleIDs_t* moduleIDs_t_new(hid_t ID16, hid_t ID8, unsigned int mod1, unsigned int mod2){
    moduleIDs_t* value = (moduleIDs_t*) malloc(sizeof(struct moduleIDs));
    //HKPackets_t* HK = (HKPackets_t*) malloc(sizeof(struct HKPackets));
    value->ID16bit = ID16;
    value->ID8bit = ID8;
    value->status = 0;
    value->mod1Name = mod1;
    value->mod2Name = mod2;
    value->next_moduleID = NULL;
    value->frameNum = -1;
    value->dataNum = -1;
    return value;
}

void HKPackets_init(HKPackets_t* data){
    for(int i = 0; i < NUMPKT; i++){
        data->BOARDLOC[i] = 0;
        data->HVMON0[i] = data->HVMON1[i] = data->HVMON2[i] = data->HVMON3[i] = -1;
        data->HVIMON0[i] = data->HVIMON1[i] = data->HVIMON2[i] = data->HVIMON3[i] = 0;
    }
}

moduleIDs_t* moduleIDs_t_new(){
    moduleIDs_t_new(0,0,-1,-1);
}

moduleIDs_t* moduleIDs_t_new(hid_t ID16, hid_t ID8, unsigned mod1){
    moduleIDs_t_new(ID16, ID8, mod1, -1);
}

moduleIDs_t* get_moduleID(moduleIDs_t* list, unsigned int ind){
    if(list != NULL && ind > 0)
        return get_moduleID(list->next_moduleID, ind-1);
    return list;
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

    H5Sclose(dataspace);
    H5Tclose(datatype);
    H5Aclose(attribute);

}

void createStrAttribute2(hid_t group, const char* name, hsize_t* dimsf, char data[NUMPKT][STRBUFFSIZE]) {
    hid_t       datatype, dataspace;   /* handles */
    hid_t       attribute;

    dataspace = H5Screate_simple(sizeof(dimsf)/sizeof(dimsf[0]), dimsf, NULL);

    datatype = H5Tcopy(H5T_C_S1);
    H5Tset_size(datatype, STRBUFFSIZE);
    H5Tset_strpad(datatype, H5T_STR_NULLTERM);
    H5Tset_cset(datatype, H5T_CSET_UTF8);

    attribute = H5Acreate(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, datatype, data[0]);

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
    sprintf(modName, "./ModulePair_%05u_%05u", mod1Name, mod2Name);

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

    return newfile;
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
        H5Gclose(head->ID16bit);
        H5Gclose(head->ID8bit);
        H5Gclose(head->ID16bitframe);
        H5Gclose(head->ID8bitframe);
        head = head->next_moduleID;
        free(currentmodule);
        currentmodule = head;
    }
}

void storeHKdata(hid_t frame, HKPackets_t* HK) {
    hid_t datatypeI, datatypeU64, datatypeU32;
    hsize_t dimsf[1];
    dimsf[0] = 2;

    datatypeI = H5Tcopy(H5T_STD_I64LE);
    datatypeU64 = H5Tcopy(H5T_STD_U64LE);
    datatypeU32 = H5Tcopy(H5T_STD_U32LE);

    createStrAttribute2(frame, "SYSTIME", dimsf, HK->SYSTIME);
    //printf("Here is thetime for 1 %s\n", HK->SYSTIME[0]);
    //printf("Here is thetime for 2 %s\n", HK->SYSTIME[1]);

    createNumAttribute2(frame, "HVMON0", H5T_STD_I64LE, dimsf, HK->HVMON0);
    createNumAttribute2(frame, "HVMON1", H5T_STD_I64LE, dimsf, HK->HVMON1);
    createNumAttribute2(frame, "HVMON2", H5T_STD_I64LE, dimsf, HK->HVMON2);
    createNumAttribute2(frame, "HVMON3", H5T_STD_I64LE, dimsf, HK->HVMON3);

    createNumAttribute2(frame, "HVIMON0", H5T_STD_U64LE, dimsf, HK->HVIMON0);
    createNumAttribute2(frame, "HVIMON1", H5T_STD_U64LE, dimsf, HK->HVIMON1);
    createNumAttribute2(frame, "HVIMON2", H5T_STD_U64LE, dimsf, HK->HVIMON2);
    createNumAttribute2(frame, "HVIMON3", H5T_STD_U64LE, dimsf, HK->HVIMON3);

    createNumAttribute2(frame, "RAWHVMON", H5T_STD_I64LE, dimsf, HK->RAWHVMON);

    createNumAttribute2(frame, "V12MON",  H5T_STD_U64LE, dimsf, HK->V12MON);
    createNumAttribute2(frame, "V18MON",  H5T_STD_U64LE, dimsf, HK->V18MON);
    createNumAttribute2(frame, "V33MON",  H5T_STD_U64LE, dimsf, HK->V33MON);
    createNumAttribute2(frame, "V37MON",  H5T_STD_U64LE, dimsf, HK->V37MON);

    createNumAttribute2(frame, "I10MON",  H5T_STD_U64LE, dimsf, HK->I10MON);
    createNumAttribute2(frame, "I18MON",  H5T_STD_U64LE, dimsf, HK->I18MON);
    createNumAttribute2(frame, "I33MON",  H5T_STD_U64LE, dimsf, HK->I33MON);

    createNumAttribute2(frame, "TEMP1",  H5T_STD_I64LE, dimsf, HK->TEMP1);
    createNumAttribute2(frame, "TEMP2",  H5T_STD_U64LE, dimsf, HK->TEMP2);

    createNumAttribute2(frame, "VCCINT",  H5T_STD_U64LE, dimsf, HK->VCCINT);
    createNumAttribute2(frame, "VCCAUX",  H5T_STD_U64LE, dimsf, HK->VCCAUX);
    createNumAttribute2(frame, "UID",  H5T_STD_U64LE, dimsf, HK->UID);

    createNumAttribute2(frame, "SHUTTER_STATUS",  H5T_STD_U8LE, dimsf, HK->SHUTTER_STATUS);
    createNumAttribute2(frame, "LIGHT_SENSOR_STATUS",  H5T_STD_U8LE, dimsf, HK->LIGHT_STATUS);

    createNumAttribute2(frame, "FWID0",  H5T_STD_U64LE, dimsf, HK->FWID0);
    createNumAttribute2(frame, "FWID1",  H5T_STD_U64LE, dimsf, HK->FWID1);

}

void fetchHKdata(HKPackets_t* HK, redisContext* redisServer) {
    redisReply *reply;
    char command[50];

    for(int i = 0; i < NUMPKT; i++){
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "SYSTIME");
        reply = (redisReply *)redisCommand(redisServer, command);
        strcpy(HK->SYSTIME[i], reply->str);
        freeReplyObject(reply);

        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "HVMON0");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->HVMON0[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "HVMON1");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->HVMON1[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "HVMON2");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->HVMON2[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "HVMON3");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->HVMON3[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);

        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "HVIMON0");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->HVIMON0[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "HVIMON1");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->HVIMON1[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "HVIMON2");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->HVIMON2[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "HVIMON3");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->HVIMON3[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);

        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "RAWHVMON");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->RAWHVMON[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);

        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "V12MON");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->V12MON[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "V18MON");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->V18MON[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "V33MON");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->V33MON[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "V37MON");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->V37MON[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);

        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "I10MON");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->I10MON[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "I18MON");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->I18MON[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "I33MON");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->I33MON[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);

        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "TEMP1");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->TEMP1[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "TEMP2");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->TEMP2[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);

        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "VCCINT");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->VCCINT[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "VCCAUX");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->VCCAUX[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);

        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "UID");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->UID[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);

        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "SHUTTER_STATUS");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->SHUTTER_STATUS[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "LIGHT_SENSOR_STATUS");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->LIGHT_STATUS[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);

        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "FWID0");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->FWID0[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
        sprintf(command, "HGET %lu %s", HK->BOARDLOC[i], "FWID1");
        reply = (redisReply *)redisCommand(redisServer, command);
        HK->FWID1[i] = strtoll(reply->str, NULL, 10);
        freeReplyObject(reply);
    }
}

int createDataBlock(moduleIDs_t* module, HKPackets_t* HouseKeeping){
    char frameName[50];
    /*if (module->frameNum > 0)
        H5Gclose(module->ID16bitframe);
        H5Gclose(module->ID8bitframe);*/

    // Create a new group of frames to store the metadata and with new sets of frame
    module->frameNum = module->frameNum + 1;
    sprintf(frameName, FRAME_FORMAT, module->frameNum);
    module->ID16bitframe = H5Gcreate(module->ID16bit, frameName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    module->ID8bitframe = H5Gcreate(module->ID8bit, frameName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Store the metadata from housekeeping into the new frame
    storeHKdata(module->ID16bitframe, HouseKeeping);
    storeHKdata(module->ID8bitframe, HouseKeeping);


    return 1;
}

void writeDataBlock(hid_t frame, uint16_t data_ptr[NUMPKT][SCIDATASIZE], int index){
    hid_t dataset;
    char name[50];

    sprintf(name, DATA_FORMAT, index);
    dataset = H5Dcreate2(frame, name, storageType, storageSpace,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    H5Dwrite(dataset, H5T_STD_U16LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_ptr);

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

void storeData(moduleIDs_t* module, char* data_ptr, uint16_t boardLoc, char acqmode){
    uint16_t* moduleData;
    hid_t moduleFrame;
    int mode;

    if(acqmode == 0x2 || acqmode == 0x3){
        moduleFrame = module->ID16bitframe;
        mode = 16;
    } else if (acqmode == 0x6 || acqmode == 0x7){
        moduleFrame = module->ID8bitframe;
        mode = 8;
    } else {
        printf("A new mode was identify acqmode=%X\n", acqmode);
        printf("packet skipped\n");
        return;
    }

    if(boardLoc == module->mod1Name){
        if (module->status == 1){
            module->dataNum = module->dataNum + 1;
            writeDataBlock(moduleFrame, module->data, module->dataNum);
            
            module->status = 0;
        }
        moduleData = module->data[0];

        storePktData(moduleData, data_ptr, mode);

        if (module->status = 2){
            module->dataNum = module->dataNum + 1;
            writeDataBlock(moduleFrame, module->data, module->dataNum);
            
            module->status = 0;
        } else {
            module->status = 1;
        }

    } else if (boardLoc == module->mod2Name){
        if (module->status == 2){
            module->dataNum = module->dataNum + 1;
            writeDataBlock(moduleFrame, module->data, module->dataNum);
            
            module->status = 0;
        }
        moduleData = module->data[1];

        storePktData(moduleData, data_ptr, mode);

        if (module->status = 1){
            module->dataNum = module->dataNum + 1;
            writeDataBlock(moduleFrame, module->data, module->dataNum);
            
            module->status = 0;
        } else {
            module->status = 1;
        }

    } else {
        //TODO Create a separate data block for this module
    }

    /*if(boardLoc == module->mod1Name){
        memcpy(module->data[0], data_ptr, SCIDATASIZE);
        
        if (module->status == 2){
            if (acqmode == 0x2 || acqmode == 0x3) {
                writeDataBlock(module->ID16bitframe, module->data);
            } else if (acqmode == 0x6 || acqmode == 0x7){
                writeDataBlock(module->ID8bitframe, module->data);
            }
            module->status = 0;
        } else {
            module->status = 1;
        }
    } else if (boardLoc == module->mod2Name) {

        memcpy(module->data[1], data_ptr, SCIDATASIZE);

        if (module->status == 1){
            if (acqmode == 0x2 || acqmode == 0x3) {
                writeDataBlock(module->ID16bitframe, module->data);
            } else if (acqmode == 0x6 || acqmode == 0x7){
                writeDataBlock(module->ID8bitframe, module->data);
            }
            module->status = 0;
        } else {
            module->status = 2;
        }
    }*/
}

moduleIDs_t* get_module_info(moduleIDs_t* list, unsigned int ind){
    if(list != NULL && ind > 0)
        return get_module_info(list->next_moduleID, ind-1);
    return list;
}

static void *run(hashpipe_thread_args_t * args){

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
    uint16_t boardLoc;
    char acqmode;

    //FILE * HSD_file;
    //HSD_file=fopen("./data.out", "w");

    
    /*Initialization of Redis Server Values*/
    printf("------------------SETTING UP REDIS ------------------\n");
    redisContext *redisServer = redisConnect("127.0.0.1", 6379);
    if (redisServer != NULL && redisServer->err){
        printf("Error: %s\n", redisServer->errstr);
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
    fileIDs_t* file;
    hid_t datatype, dataspace, dataset;   /* handles */
    hsize_t dimsf[2];
    moduleIDs_t* moduleListBegin = moduleIDs_t_new();
    moduleIDs_t* moduleListEnd = moduleListBegin;
    moduleIDs_t* currentModule;
    unsigned int moduleListSize = 1;
    unsigned int moduleInd[0xffff];
    memset(moduleInd, -1, sizeof(moduleInd));
    int data[2][PKTSIZE];

    char fileName[100];

    time_t t = time(NULL);
    struct tm tm = *gmtime(&t);
    char currTime[100];

    FILE *modConfig_file;
    char fbuf[100];
    char cbuf;
    unsigned int mod1Name;
    unsigned int mod2Name;



    modConfig_file = fopen(CONFIGFILE, "r");
    if (modConfig_file == NULL) {
        perror("Error Opening File\n");
        return(NULL);
    }

    dimsf[0] = NUMPKT;
    dimsf[1] = PKTSIZE;
    dataspace = H5Screate_simple(RANK, dimsf, NULL);
    datatype = H5Tcopy(H5T_STD_U64LE);
    
    sprintf(currTime, "%04i-%02i-%02iT%02i:%02i:%02i UTC",tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    sprintf(fileName, H5FILE_NAME_FORMAT, OBSERVATORY, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    
    file = createNewFile(fileName, currTime);

    cbuf = getc(modConfig_file);
    while(cbuf != EOF){
        ungetc(cbuf, modConfig_file);
        if (cbuf != '#'){
            if (fscanf(modConfig_file, "%u %u\n", &mod1Name, &mod2Name) == 2){
                if (moduleInd[mod1Name] == -1 && moduleInd[mod2Name] == -1){
                    moduleInd[mod1Name] = moduleInd[mod2Name] = moduleListSize;
                    moduleListSize++;

                    printf("Created Module Pair: %u.%u and %u.%u\n", (unsigned int) mod1Name/0x100, mod1Name % 0x100, mod2Name/0x100, mod2Name % 0x100);

                    moduleListEnd->next_moduleID = moduleIDs_t_new(createModPair(file->bit16IMGData, mod1Name, mod2Name), 
                                                                    createModPair(file->bit8IMGData, mod1Name, mod2Name), 
                                                                    mod1Name, mod2Name);
                    
                    moduleListEnd = moduleListEnd->next_moduleID;
                }
            }
        } else {
            if (fgets(fbuf, 100, modConfig_file) == NULL){
                break;
            }
        }
        cbuf = getc(modConfig_file);
    }

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

        for(int i = 0; i < N_PKT_PER_BLOCK; i++){
            boardLoc = ((block_ptr[i*PKTSIZE+5] << 8) & 0xff00) | ((block_ptr[i*PKTSIZE+4]) & 0x00ff);
            acqmode = block_ptr[i*PKTSIZE];

            if (moduleInd[boardLoc] == -1){
                moduleInd[boardLoc] = moduleListSize;
                moduleListSize++;

                printf("Detected New Module not in Config File: %u.%u\n", (unsigned int) boardLoc/0x100, boardLoc % 0x100);

                moduleListEnd->next_moduleID = moduleIDs_t_new(createModPair(file->bit16IMGData, boardLoc, 0), 
                                                                    createModPair(file->bit8IMGData, boardLoc, 0), 
                                                                    boardLoc, 0);
                moduleListEnd = moduleListEnd->next_moduleID;
                currentModule = moduleListEnd;

            } else {
                currentModule = get_module_info(moduleListBegin, moduleInd[boardLoc]);
            }
            //printf("Current Module Pair: %i, %i\n", currentModule->mod1Name, currentModule->mod2Name);

            sprintf(command, "HGET UPDATED %i", currentModule->mod1Name);
            reply = (redisReply *)redisCommand(redisServer, command);
            if (strtol(reply->str, NULL, 10) == 1){
                freeReplyObject(reply);
                sprintf(command, "HGET UPDATED %i", currentModule->mod2Name);
                reply = (redisReply *)redisCommand(redisServer, command);

                if (strtol(reply->str, NULL, 10) == 1){

                    HK->BOARDLOC[0] = currentModule->mod1Name;
                    HK->BOARDLOC[1] = currentModule->mod2Name;
                    fetchHKdata(HK, redisServer);

                    createDataBlock(currentModule, HK);

                    currentModule->dataNum = -1;

                    freeReplyObject(reply);
                    sprintf(command, "HSET UPDATED %i 0", currentModule->mod1Name);
                    reply = (redisReply *)redisCommand(redisServer, command);
                    freeReplyObject(reply);
                    sprintf(command, "HSET UPDATED %i 0", currentModule->mod2Name);
                    reply = (redisReply *)redisCommand(redisServer, command);

                }
            }
            freeReplyObject(reply);

            storeData(currentModule, block_ptr + (i*PKTSIZE+16), boardLoc, acqmode);
        }



        HSD_output_databuf_set_free(db,block_idx);
	    block_idx = (block_idx + 1) % db->header.n_block;
	    mcnt++;

        //Will exit if thread has been cancelled
        pthread_testcancel();

    }

    closeFile(file);
    H5Sclose(dataspace);
    H5Tclose(datatype);
    fclose(modConfig_file);
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
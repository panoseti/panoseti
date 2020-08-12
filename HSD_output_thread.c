/*
 * demo1_output_thread.c
 */

#include <stdio.h>
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
#define CONFIGFILE "./modulePair.config"

typedef struct fileIDs {
    hid_t       file;         /* file and dataset handles */
    hid_t       bit16IMGData, bit8IMGData, PHData, ShortTransient, bit16HCData, bit8HCData, DynamicMeta;
} fileIDs_t;

typedef struct moduleIDs {
    hid_t name;
    moduleIDs* next_moduleID;
} moduleIDs_t;

moduleIDs_t* moduleIDs_t_new(){
    moduleIDs_t* value = (moduleIDs_t*) malloc(sizeof(struct moduleIDs));
    value->name = 0;
    value->next_moduleID = NULL;
    return value;
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

hid_t createModPair(hid_t group, unsigned int mod1Name, unsigned int mod2Name) {
    hid_t   modulePair, moduleInfo;
    char    modName[50];
    sprintf(modName, "./ModulePair_%05u_%05u", mod1Name, mod2Name);

    modulePair = H5Gcreate(group, modName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    createNumAttribute(modulePair, "Module1", H5T_STD_U64LE , mod1Name);
    createNumAttribute(modulePair, "Module2", H5T_STD_U64LE , mod2Name);

    return modulePair;//H5Gclose(modulePair);

}

fileIDs_t createNewFile(char* fileName, char* currTime){
    fileIDs_t newfile;

    newfile.file = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    createStrAttribute(newfile.file, "dateCreated", currTime);
    newfile.bit16IMGData = H5Gcreate(newfile.file, "/bit16IMGData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile.bit8IMGData = H5Gcreate(newfile.file, "/bit8IMGData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile.PHData = H5Gcreate(newfile.file, "/PHData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile.ShortTransient = H5Gcreate(newfile.file, "/ShortTransient", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile.bit16HCData =  H5Gcreate(newfile.file, "/bit16HCData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    newfile.bit8HCData = H5Gcreate(newfile.file, "/bit8HCData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 

    return newfile;
}

void closeFile(fileIDs_t file){
    H5Fclose(file.file);
    H5Gclose(file.bit16IMGData);
    H5Gclose(file.bit8IMGData);
    H5Gclose(file.PHData);
    H5Gclose(file.ShortTransient);
}

void writeDataBlock(){


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
    //FILE * HSD_file;
    //HSD_file=fopen("./data.out", "w");

    
    /*Initialization of Redis Server Values*/
    printf("----------SETTING UP REDIS ----------\n");
    redisContext *redisServer = redisConnect("127.0.0.1", 6379);
    if (redisServer != NULL && redisServer->err){
        printf("Error: %s\n", redisServer->errstr);
    } else {
        printf("Connect to Redis\n");
    }
    redisReply *keysReply;
    redisReply *reply;
    // Uncomment following lines for redis servers with password
    // reply = redisCommand(redisServer, "AUTH password");
    // freeReplyObject(reply);


    /* Initialization of HDF5 Values*/
    printf("-----------SETTING UP HDF5 ----------\n");
    fileIDs_t file;
    hid_t datatype, dataspace, dataset;   /* handles */
    hsize_t dimsf[2];
    moduleIDs_t* moduleListBegin = moduleIDs_t_new();
    moduleIDs_t* moduleListEnd = moduleListBegin;
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

    dimsf[0] = RANK;
    dimsf[1] = PKTSIZE;
    dataspace = H5Screate_simple(RANK, dimsf, NULL);
    datatype = H5Tcopy(H5T_STD_U64LE);
    
    sprintf(currTime, "%04i_%02i_%02i_%02i-%02i-%02i UTC",tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    sprintf(fileName, H5FILE_NAME_FORMAT, OBSERVATORY, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    
    file = createNewFile(fileName, currTime);

    cbuf = getc(modConfig_file);
    while(cbuf != EOF){
        ungetc(cbuf, modConfig_file);
        if (cbuf != '#'){
            if (fscanf(modConfig_file, "%u %u\n", &mod1Name, &mod2Name) == 2){
                H5Gclose(createModPair(file.bit16IMGData, mod1Name, mod2Name));
                H5Gclose(createModPair(file.bit8IMGData, mod1Name, mod2Name));
                if (moduleInd[mod1Name] == -1 && moduleInd[mod2Name] == -1){
                    moduleInd[mod1Name] = moduleInd[mod2Name] = moduleListSize;
                    moduleListSize++;

                    printf("Created Module Pair: %u.%u and %u.%u\n", (unsigned int) mod1Name/0x100, mod1Name % 0x100, mod2Name/0x100, mod2Name % 0x100);

                    moduleListEnd->next_moduleID = moduleIDs_t_new();
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
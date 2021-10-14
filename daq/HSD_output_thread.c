/* HSD_output_thread.c
 *
 * Writes the data to HDF5 output file
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include "hashpipe.h"
#include "HSD_databuf.h"
#include "hiredis/hiredis.h"
#include "../util/pff.cpp"
#include "../util/dp.h"

//Defining the names of redis keys and files
#define OBSERVATORY "LICK"
#define GPSPRIMKEY "GPSPRIM"
#define GPSSUPPKEY "GPSSUPP"
#define WRSWITCHKEY "WRSWITCH"
#define UPDATEDKEY "UPDATED"

////////// Structures for Reading and Parsing file in PFF////////////////

struct PF {
    DATA_PRODUCT dataProduct;
    FILE *filePtr;
    PF(FILENAME_INFO *fileInfo, DIRNAME_INFO *dirInfo);
    PF(const char *dirName, const char *fileName);
};

struct FILE_PTRS{
    DIRNAME_INFO dir_info;
    FILENAME_INFO file_info;
    FILE *dynamicMeta, *bit16Img, *bit8Img, *PHImg;
    FILE_PTRS(const char *diskDir, DIRNAME_INFO *dirInfo, FILENAME_INFO *fileInfo, const char *file_mode);
    void make_files(const char *diskDir, const char *file_mode);
};

FILE_PTRS::FILE_PTRS(const char *diskDir, DIRNAME_INFO *dirInfo, FILENAME_INFO *fileInfo, const char *file_mode){
    string fileName;
    string dirName;
    dirInfo->make_dirname(dirName);
    dirName = diskDir + dirName + "/";
    mkdir(dirName.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    

    for (int dp = DP_DYNAMIC_META; dp <= DP_PH_IMG; dp++){
        fileInfo->data_product = (DATA_PRODUCT)dp;
        fileInfo->make_filename(fileName);
        switch (dp){
            case DP_DYNAMIC_META:
                this->dynamicMeta = fopen((dirName + fileName).c_str(), file_mode);
                break;
            case DP_BIT16_IMG:
                this->bit16Img = fopen((dirName + fileName).c_str(), file_mode);
                break;
            case DP_BIT8_IMG:
                this->bit8Img = fopen((dirName + fileName).c_str(), file_mode);
                break;
            case DP_PH_IMG:
                this->PHImg = fopen((dirName + fileName).c_str(), file_mode);
                break;
            default:
                break;
        }
        if (access(dirName.c_str(), F_OK) == -1) {
            printf("Error: Unable to access file - %s\n", dirName.c_str());
            exit(0);
        }
        printf("Created file %s\n", (dirName + fileName).c_str());
    }
}

void FILE_PTRS::make_files(const char *diskDir, const char *file_mode){
    string fileName;
    string dirName;
    this->dir_info.make_dirname(dirName);
    dirName = diskDir + dirName + "/";
    mkdir(dirName.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    

    for (int dp = DP_DYNAMIC_META; dp <= DP_PH_IMG; dp++){
        this->file_info.data_product = (DATA_PRODUCT)dp;
        this->file_info.make_filename(fileName);
        switch (dp){
            case DP_DYNAMIC_META:
                this->dynamicMeta = fopen((dirName + fileName).c_str(), file_mode);
                break;
            case DP_BIT16_IMG:
                this->bit16Img = fopen((dirName + fileName).c_str(), file_mode);
                break;
            case DP_BIT8_IMG:
                this->bit8Img = fopen((dirName + fileName).c_str(), file_mode);
                break;
            case DP_PH_IMG:
                this->PHImg = fopen((dirName + fileName).c_str(), file_mode);
                break;
            default:
                break;
        }
        if (access(dirName.c_str(), F_OK) == -1) {
            printf("Error: Unable to access file - %s\n", dirName.c_str());
            exit(0);
        }
        printf("Created file %s\n", (dirName + fileName).c_str());
    }
}


static char config_location[STRBUFFSIZE];

static char save_location[STRBUFFSIZE];
static long long file_size = 0;
static long long max_file_size = 0; //IN UNITS OF APPROX 2 BYTES OR 16 bits


static redisContext *redis_server;
static FILE_PTRS *data_files[MODULEINDEXSIZE] = {NULL};
static FILE *dynamic_meta;


FILE_PTRS *data_file_init(const char *diskDir, int dome, int module) {
    time_t t = time(NULL);

    DIRNAME_INFO dirInfo(t, OBSERVATORY);
    FILENAME_INFO filenameInfo(t, DP_STATIC_META, 0, dome, module, 0);
    return new FILE_PTRS(diskDir, &dirInfo, &filenameInfo, "w");
}

int write_img_header_file(FILE *fileToWrite, HSD_output_block_header_t *dataHeader, int blockIndex){
    fprintf(fileToWrite, "{ ");
    for (int i = 0; i < QUABOPERMODULE; i++){
        fprintf(fileToWrite,
        "quabo %u: { acq_mode: %u, mod_num: %u, qua_num: %u, pkt_num : %u, pkt_nsec : %u, tv_sec : %li, tv_usec : %li, status : %u}",
        i,
        dataHeader->img_pkt_head[blockIndex].pkt_head[i].acq_mode,
        dataHeader->img_pkt_head[blockIndex].pkt_head[i].mod_num,
        dataHeader->img_pkt_head[blockIndex].pkt_head[i].qua_num,
        dataHeader->img_pkt_head[blockIndex].pkt_head[i].pkt_num,
        dataHeader->img_pkt_head[blockIndex].pkt_head[i].pkt_nsec,
        dataHeader->img_pkt_head[blockIndex].pkt_head[i].tv_sec,
        dataHeader->img_pkt_head[blockIndex].pkt_head[i].tv_usec,
        dataHeader->img_pkt_head[blockIndex].status[i]
        );
        if (i < QUABOPERMODULE-1){
            fprintf(fileToWrite, ", ");
        }
    }
    fprintf(fileToWrite, "}");
}

int write_module_img_file(HSD_output_block_t *dataBlock, int blockIndex){
    FILE *fileToWrite;
    FILE_PTRS *moduleToWrite;
    int mode = dataBlock->header.img_pkt_head[blockIndex].mode;
    int modSizeMultiplier = mode/8;

    moduleToWrite = data_files[dataBlock->header.img_pkt_head[blockIndex].mod_num];
    if (mode == 16) {
        fileToWrite = moduleToWrite->bit16Img;
    } else if (mode == 8){
        fileToWrite = moduleToWrite->bit8Img;
    } else {
        printf("Mode %i not recognized\n", mode);
        printf("Module Header Value\n%s", dataBlock->header.img_pkt_head[blockIndex].toString().c_str());
        return 0;
    }
    
    if (moduleToWrite == NULL){
        printf("Module To Write is null\n");
        return 0;
    } else if (fileToWrite == NULL){
        printf("File to Write is null\n");
        return 0;
    } 

    pff_start_json(fileToWrite);

    write_img_header_file(fileToWrite, &(dataBlock->header), blockIndex);

    pff_end_json(fileToWrite);

    pff_write_image(fileToWrite, 
        QUABOPERMODULE*SCIDATASIZE*modSizeMultiplier, 
        dataBlock->stream_block + (blockIndex*MODULEDATASIZE));
    return 1;
}

/**
 * Given a file ptr object and the output datablock with its index it will write the image data
 * to its corresponding data file in the PFF format
 * @param currentFilePtrs The current file pointers for the given module
 * @param dataBlock The datablock that is currently being read from the output buffer
 * @param frameIndex The index of the datablock to read the image frame from
 */
int write_img_files(HSD_output_block_t *dataBlock, int blockIndex){
    write_module_img_file(dataBlock, blockIndex);
}

int create_data_files(){
    FILE *configFile = fopen(config_location, "r");
    char fbuf[STRBUFFSIZE];
    char cbuf;
    unsigned int modNum;

    if (configFile == NULL) {
        perror("Error Opening Config File");
        exit(1);
    }

    cbuf = getc(configFile);

    while (cbuf != EOF){
        ungetc(cbuf, configFile);
        if (cbuf != '#') {
            if (fscanf(configFile, "%u\n", &modNum) == 1){
                if (data_files[modNum] == NULL) {
                    data_files[modNum] = data_file_init(save_location, 0, modNum);
                    printf("Created Data file for Module %u\n", modNum);
                }
            }
        } else {
            if (fgets(fbuf, STRBUFFSIZE, configFile) == NULL) {
                break;
            }
        }
        cbuf = getc(configFile);
    }

    if (fclose(configFile) == EOF) {
        printf("Warning: Unable to close module configuration file.\n");
    }
}

void write_redis_key(redisContext *redisServer, const char *key, FILE *filePtr){
    redisReply *reply = (redisReply *)redisCommand(redisServer, "HGETALL %s", key);
    if (reply->type != REDIS_REPLY_ARRAY){
        printf("Warning: Unable to get %s keys from Reids. Skipping Redis values from %s.", key, key);
        return;
    }
    pff_start_json(filePtr);
    fprintf(filePtr, "{ RedisKey :%s", key);
    for (int i = 0; i < reply->elements; i=i+2){
        fprintf(filePtr, ", %s :%s", reply->element[i]->str, reply->element[i+1]->str);
    }
    fprintf(filePtr, "}");
    pff_end_json(filePtr);
}

void check_redis(redisContext *redisServer){
    redisReply *reply = (redisReply *)redisCommand(redisServer, "HGETALL %s", UPDATEDKEY);
    if (reply->type != REDIS_REPLY_ARRAY){
        printf("Warning: Unable to get Updated keys from Redis. Skipping Redis values.\n");
        freeReplyObject(reply);
        return;
    }
    for (int i = 0; i < reply->elements; i=i+2){
        if (strcmp(reply->element[i+1]->str, "0") == 0){continue;}

        if (isdigit(reply->element[i]->str[0])){
            if (data_files[strtol(reply->element[i]->str, NULL, 10) >> 2] != NULL){
                write_redis_key(redisServer, 
                    reply->element[i]->str, 
                    data_files[strtol(reply->element[i]->str, NULL, 10) >> 2]->dynamicMeta);
            }
        } else {
            write_redis_key(redisServer, reply->element[i]->str, dynamic_meta);
        }
    } 
}

//Signal handeler to allow for hashpipe to exit gracfully and also to allow for creating of new files by command.
static int QUITSIG;

void QUIThandler(int signum) {
    QUITSIG = 1;
}

static int init(hashpipe_thread_args_t *args)
{
    // Get info from status buffer if present
    hashpipe_status_t st = args->st;
    printf("\n\n-----------Start Setup of Output Thread--------------\n");
    sprintf(save_location, DATAFILE_DEFAULT);
    hgets(st.buf, "SAVELOC", STRBUFFSIZE, save_location);
    if (save_location[strlen(save_location) - 1] != '/') {
        char endingSlash = '/';
        strncat(save_location, &endingSlash, 1);
    }
    printf("Save Location: %s\n", save_location);

    sprintf(config_location, CONFIGFILE_DEFAULT);
    hgets(st.buf, "CONFIG", STRBUFFSIZE, config_location);
    printf("Config Location: %s\n", config_location);

    int maxSizeInput = 0;

    hgeti4(st.buf, "MAXFILESIZE", &maxSizeInput);
    max_file_size = maxSizeInput * 2E6;

    /*Initialization of Redis Server Values*/
    printf("------------------SETTING UP REDIS ------------------\n");
    redis_server = redisConnect("127.0.0.1", 6379);
    int attempts = 0;
    while (redis_server != NULL && redis_server->err) {
        printf("Error: %s\n", redis_server->errstr);
        attempts++;
        if (attempts >= 12) {
            printf("Unable to connect to Redis.\n");
            exit(0);
        }
        printf("Attempting to reconnect in 5 seconds.\n");
        sleep(5);
        redis_server = redisConnect("127.0.0.1", 6379);
    }

    printf("Connected to Redis\n");
    redisReply *keysReply;
    redisReply *reply;
    // Uncomment following lines for redis servers with password
    // reply = redisCommand(redis_server, "AUTH password");
    // freeReplyObject(reply);

    printf("\n---------------SETTING UP DATA File------------------\n");
    time_t t = time(NULL);

    DIRNAME_INFO dirInfo(t, OBSERVATORY);
    string dirName;
    dirInfo.make_dirname(dirName);
    dirName = save_location + dirName + "/";
    mkdir(dirName.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    printf("Creating file : %s\n", (dirName + "dynamic_meta.pff").c_str());
    dynamic_meta = fopen((dirName + "dynamic_meta.pff").c_str(), "w");
    create_data_files();
    check_redis(redis_server);
    printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n");
    printf("-----------Finished Setup of Output Thread-----------\n\n");    

    return 0;
}

void close_all_resources() {
    for (int i = 0; i < MODULEINDEXSIZE; i++){
        if (data_files[i] != NULL){
            fclose(data_files[i]->dynamicMeta);
            fclose(data_files[i]->bit16Img);
            fclose(data_files[i]->bit8Img);
            fclose(data_files[i]->PHImg);
        }
    }
}

static void *run(hashpipe_thread_args_t *args) {

    signal(SIGQUIT, QUIThandler);
    QUITSIG = 0;

    //FETCH STATIC AND DYNAMIC REDIS DATA

    printf("---------------Running Output Thread-----------------\n\n");

    /*Initialization of HASHPIPE Values*/
    // Local aliases to shorten access to args fields
    // Our input buffer happens to be a demo1_ouput_databuf
    HSD_output_databuf_t *db = (HSD_output_databuf_t *)args->ibuf;
    hashpipe_status_t st = args->st;
    const char *status_key = args->thread_desc->skey;

    int rv;
    int block_idx = 0;
    uint64_t mcnt = 0;
    FILE_PTRS *currentDataFile;

    /* Main loop */
    while (run_threads()) {

        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "OUTBLKIN", block_idx);
        hputi8(st.buf, "OUTMCNT", mcnt);
        hputs(st.buf, status_key, "waiting");
        hashpipe_status_unlock_safe(&st);

        //Wait for the output buffer to be free
        while ((rv = HSD_output_databuf_wait_filled(db, block_idx)) != HASHPIPE_OK)
        {
            if (rv == HASHPIPE_TIMEOUT)
            {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked");
                hashpipe_status_unlock_safe(&st);
                continue;
            }
            else
            {
                hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
                pthread_exit(NULL);
                break;
            }
        }

        // Mark the buffer as processing
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "processing");
        hashpipe_status_unlock_safe(&st);

        //TODO FETCH AND STORE DYNAMIC METATDATA
        //STORE FRAMES FROM OUTPUT BUFFER ONTO DATAFILES
        check_redis(redis_server);
        for (int i = 0; i < db->block[block_idx].header.stream_block_size; i++){
            write_img_files(&(db->block[block_idx]), i);
        }

        if (QUITSIG || file_size > max_file_size) {
            printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n\n");
            file_size = 0;
            QUITSIG = 0;
        }

        //TODO check mcnt
        if (db->block[block_idx].header.INTSIG) {
            close_all_resources();
            printf("OUTPUT_THREAD Ended\n");
            break;
        }

        HSD_output_databuf_set_free(db, block_idx);
        block_idx = (block_idx + 1) % db->header.n_block;
        mcnt++;

        /* Term conditions */

        //Will exit if thread has been cancelled
        pthread_testcancel();
    }

    printf("Returned Output_thread\n");
    return THREAD_OK;
}

/**
 * Sets the functions and buffers for this thread
 */
static hashpipe_thread_desc_t HSD_output_thread = {
    name : "HSD_output_thread",
    skey : "OUTSTAT",
    init : init,
    run : run,
    ibuf_desc : {HSD_output_databuf_create},
    obuf_desc : {NULL}
};

static __attribute__((constructor)) void ctor()
{
    register_hashpipe_thread(&HSD_output_thread);
}

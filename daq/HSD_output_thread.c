/* HSD_output_thread.c
 *
 * Writes the data to HDF5 output file
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
#include <string>
#include "hashpipe.h"
#include "HSD_databuf.h"
#include "hiredis/hiredis.h"
#include "../util/pff.cpp"

//Defining the names of redis keys and files
#define OBSERVATORY "LICK"
#define GPSPRIMNAME "GPSPRIM"
#define GPSSUPPNAME "GPSSUPP"
#define WRSWITCHNAME "WRSWITCH"

////////// Structures for Reading and Parsing file in PFF////////////////

struct PF {
    DATA_PRODUCT dataProduct;
    FILE *filePtr;
    PF(FILENAME_INFO *fileInfo, DIRNAME_INFO *dirInfo);
    PF(const char *dirName, const char *fileName);
};


FILE_PTRS::FILE_PTRS(const char *diskDir, DIRNAME_INFO *dirInfo, FILENAME_INFO *fileInfo, const char *mode){
    string fileName;
    string dirName;
    dirInfo->make(dirName);
    dirName = diskDir + dirName + "/";
    mkdir(dirName.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    

    for (int dp = DP_DYNAMIC_META; dp <= DP_PH_IMG; dp++){
        fileInfo->data_product = (DATA_PRODUCT)dp;
        fileInfo->make(fileName);
        switch (dp){
            case DP_DYNAMIC_META:
                dynamicMeta = fopen((dirName + fileName).c_str(), mode);
                break;
            case DP_BIT16_IMG:
                bit16Img = fopen((dirName + fileName).c_str(), mode);
                break;
            case DP_BIT8_IMG:
                bit8Img = fopen((dirName + fileName).c_str(), mode);
                break;
            case DP_PH_IMG:
                PHImg = fopen((dirName + fileName).c_str(), mode);
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


static char configLocation[STRBUFFSIZE];

static char saveLocation[STRBUFFSIZE];
static long long fileSize = 0;
static long long maxFileSize = 0; //IN UNITS OF APPROX 2 BYTES OR 16 bits


static redisContext *redisServer;
static FILE_PTRS *dataFiles[MODULEINDEXSIZE] = {NULL};


FILE_PTRS *data_file_init(const char *diskDir, int dome, int module) {
    time_t t = time(NULL);

    DIRNAME_INFO dirInfo(t, OBSERVATORY);
    FILENAME_INFO filenameInfo(t, DP_STATIC_META, 0, dome, module, 0);
    return new FILE_PTRS(diskDir, &dirInfo, &filenameInfo, "w");
}

int write_img_header_file(FILE *fileToWrite, HSD_output_block_header_t *dataHeader, int blockIndex, int moduleIndex){
    fprintf(fileToWrite,
    " { pktNum : %u, pktNSEC : %u, tv_sec : %li, tv_usec : %li, status : %u}",
    dataHeader->pktNum[blockIndex * PKTPERPAIR + moduleIndex],
    dataHeader->pktNSEC[blockIndex * PKTPERPAIR + moduleIndex],
    dataHeader->tv_sec[blockIndex * PKTPERPAIR + moduleIndex],
    dataHeader->tv_usec[blockIndex * PKTPERPAIR + moduleIndex],
    ((dataHeader->status[blockIndex * PKTPERPAIR + moduleIndex]) | 0x0F) >> 4*moduleIndex
    );
}

int write_module_img_file(HSD_output_block_t *dataBlock, int blockIndex, int moduleIndex){
    FILE *fileToWrite;
    FILE_PTRS *moduleToWrite;
    int mode = dataBlock->header.acqmode[blockIndex];
    int modeOffset = mode/8;

    moduleToWrite = dataFiles[dataBlock->header.modNum[blockIndex * 2 + moduleIndex]];
    if (mode == 16) {
        fileToWrite = moduleToWrite->bit16Img;
    } else if (mode == 8){
        fileToWrite = moduleToWrite->bit8Img;
    } else {
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

    write_img_header_file(fileToWrite, &(dataBlock->header), blockIndex, moduleIndex);

    pff_end_json(fileToWrite);

    pff_write_image(fileToWrite, 
        QUABOPERMODULE*SCIDATASIZE*modeOffset, 
        dataBlock->stream_block + (blockIndex*MODPAIRDATASIZE) + (moduleIndex*SCIDATASIZE*modeOffset));
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
    write_module_img_file(dataBlock, blockIndex, 0);
    write_module_img_file(dataBlock, blockIndex, 1);
}

int create_data_files(){
    FILE *configFile = fopen(configLocation, "r");
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
                if (dataFiles[modNum] == NULL) {
                    dataFiles[modNum] = data_file_init(saveLocation, 0, modNum);
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
    sprintf(saveLocation, DATAFILE_DEFAULT);
    hgets(st.buf, "SAVELOC", STRBUFFSIZE, saveLocation);
    if (saveLocation[strlen(saveLocation) - 1] != '/') {
        char endingSlash = '/';
        strncat(saveLocation, &endingSlash, 1);
    }
    printf("Save Location: %s\n", saveLocation);

    sprintf(configLocation, CONFIGFILE_DEFAULT);
    hgets(st.buf, "CONFIG", STRBUFFSIZE, configLocation);
    printf("Config Location: %s\n", configLocation);

    int maxSizeInput = 0;

    hgeti4(st.buf, "MAXFILESIZE", &maxSizeInput);
    maxFileSize = maxSizeInput * 2E6;

    /*Initialization of Redis Server Values*/
    printf("------------------SETTING UP REDIS ------------------\n");
    redisServer = redisConnect("127.0.0.1", 6379);
    int attempts = 0;
    while (redisServer != NULL && redisServer->err) {
        printf("Error: %s\n", redisServer->errstr);
        attempts++;
        if (attempts >= 12) {
            printf("Unable to connect to Redis.\n");
            exit(0);
        }
        printf("Attempting to reconnect in 5 seconds.\n");
        sleep(5);
        redisServer = redisConnect("127.0.0.1", 6379);
    }

    printf("Connected to Redis\n");
    redisReply *keysReply;
    redisReply *reply;
    // Uncomment following lines for redis servers with password
    // reply = redisCommand(redisServer, "AUTH password");
    // freeReplyObject(reply);

    printf("\n---------------SETTING UP DATA File------------------\n");
    create_data_files();
    printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n");
    printf("-----------Finished Setup of Output Thread-----------\n\n");    

    return 0;
}

void close_all_resources() {
    for (int i = 0; i < MODULEINDEXSIZE; i++){
        if (dataFiles[i] != NULL){
            fclose(dataFiles[i]->dynamicMeta);
            fclose(dataFiles[i]->bit16Img);
            fclose(dataFiles[i]->bit8Img);
            fclose(dataFiles[i]->PHImg);
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
        for (int i = 0; i < db->block[block_idx].header.stream_block_size; i++){
            write_img_files(&(db->block[block_idx]), i);
        }

        if (QUITSIG || fileSize > maxFileSize) {
            printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n\n");
            fileSize = 0;
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
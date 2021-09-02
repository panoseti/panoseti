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

static char saveLocation[STRBUFFSIZE];
static long long fileSize = 0;
static long long maxFileSize = 0; //IN UNITS OF APPROX 2 BYTES OR 16 bits

typedef struct file_ptrs{
    FILE *dynamicMeta, *bit16Img, *bit8Img, *PHImg;
} file_ptrs_t;

static redisContext* redisServer;
static file_ptrs_t dataFiles;

int data_file_init(DATA_PRODUCT dataProduct, int dome, int module) {
    time_t t = time(NULL);

    DIRNAME_INFO dirInfo(t, "LICK");
    FILENAME_INFO filenameInfo(t, dataProduct, 0, dome, module, 0);
    
    string dirName;
    string fileName;
    dirInfo.make(dirName);
    printf("Directory is :%s\n", dirName.c_str());
    filenameInfo.make(fileName);
    printf("Filename is :%s\n", fileName.c_str());

}

int fetch_storeGPSSupp(redisContext *redisServer) {
    
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
    sprintf(saveLocation, "./");
    hgets(st.buf, "SAVELOC", STRBUFFSIZE, saveLocation);
    if (saveLocation[strlen(saveLocation) - 1] != '/') {
        char endingSlash = '/';
        strncat(saveLocation, &endingSlash, 1);
        //saveLocation[strlen(saveLocation)] = '/';
    }
    printf("Save Location: %s\n", saveLocation);

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

    printf("-----------------SETTING UP DATA FILES---------------\n");
    for (int i = DP_STATIC_META; i <= DP_PH_IMG; i++){
        data_file_init((DATA_PRODUCT) i, 0, 0);
    }

    printf("-----------Finished Setup of Output Thread-----------\n\n");    

    return 0;
}

static void *run(hashpipe_thread_args_t *args) {

    signal(SIGQUIT, QUIThandler);
    QUITSIG = 0;

    printf("\n---------------SETTING UP DATA File------------------\n");

    //FETCH STATIC AND DYNAMIC REDIS DATA
    //TODO INITIALIZE DATAFILE AND STORE AT SAVELOCATION

    printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n");

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
        
        if (QUITSIG || fileSize > maxFileSize) {
            printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n\n");
            fileSize = 0;
            QUITSIG = 0;
        }

        //TODO check mcnt
        if (db->block[block_idx].header.INTSIG) {
            //closeAllResources();
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
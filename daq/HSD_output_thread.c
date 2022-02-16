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
#include <sys/types.h>
#include <string>
#include "hashpipe.h"
#include "HSD_databuf.h"
#include "../util/pff.cpp"
#include "../util/dp.h"


/**
 * Structures for Reading and Parsing file in PFF
 */
struct PF {
    DATA_PRODUCT dataProduct;
    FILE *filePtr;
    PF(FILENAME_INFO *fileInfo, DIRNAME_INFO *dirInfo);
    PF(const char *dirName, const char *fileName);
};

/**
 * Structure for storing file pointers opened by output thread.
 * A file is create for all possible data products described by pff.h
 * @see ../utls/pff.sh
 */
struct FILE_PTRS{
    DIRNAME_INFO dir_info;
    FILENAME_INFO file_info;
    FILE *dynamicMeta, *bit16Img, *bit8Img, *PHImg;
    FILE_PTRS(const char *diskDir, FILENAME_INFO *fileInfo, const char *file_mode);
    void make_files(const char *diskDir, const char *file_mode);
    void new_dp_file(DATA_PRODUCT dp, const char *diskDir, const char *file_mode);
    int increment_seqno();
};

/**
 * Constructor for file pointer structure
 * @param diskDir directory used for writing all files monitored by file pointer
 * @param fileInfo file information structure stored by file pointer
 * @param file_mode file editing mode for all files within file pointer
 */
FILE_PTRS::FILE_PTRS(const char *diskDir, FILENAME_INFO *fileInfo, const char *file_mode){
    fileInfo->copy_to(&(this->file_info));
    this->make_files(diskDir, file_mode);
}

/**
 * Creating files for the file pointer stucture given a directory.
 * @param diskDir directory for where the files will be created by file pointers
 * @param file_mode file editing mode for the new file created
 */
void FILE_PTRS::make_files(const char *diskDir, const char *file_mode){
    string fileName;
    string dirName;
    dirName = diskDir;
    
    for (int dp = DP_BIT16_IMG; dp <= DP_PH_IMG; dp++){
        this->file_info.data_product = (DATA_PRODUCT)dp;
        this->file_info.make_filename(fileName);
        switch (dp){
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
            printf("Error: Unable to access directory - %s\n", dirName.c_str());
            exit(0);
        }
        printf("Created file %s\n", (dirName + fileName).c_str());
    }
}

/**
 * Create a new file for a specified data product within file structure.
 * Method is used usually when a certain data product file has reached max file size.
 * @param dp Data product of the file that needs to be created.
 * @param diskDir Disk directory for the file pointer.
 * @param file_mode File mode of the new file created.
 */
void FILE_PTRS::new_dp_file(DATA_PRODUCT dp, const char *diskDir, const char *file_mode){
    string fileName;
    string dirName;
    dirName = diskDir;

    this->file_info.data_product = (DATA_PRODUCT)dp;
    this->file_info.start_time = time(NULL);
    this->file_info.make_filename(fileName);

    switch (dp){
        case DP_BIT16_IMG:
            fclose(this->bit16Img);
            this->bit16Img = fopen((dirName + fileName).c_str(), file_mode);
            break;
        case DP_BIT8_IMG:
            fclose(this->bit8Img);
            this->bit8Img = fopen((dirName + fileName).c_str(), file_mode);
            break;
        case DP_PH_IMG:
            fclose(this->PHImg);
            this->PHImg = fopen((dirName + fileName).c_str(), file_mode);
            break;
        default:
            break;
    }
    if (access(dirName.c_str(), F_OK) == -1) {
        printf("Error: Unable to access directory - %s\n", dirName.c_str());
        exit(0);
    }
    printf("Created file %s\n", (dirName + fileName).c_str());
}

/**
 * Increments the seqno for the filename of new files
 * @return 1 if it was successful and return 0 if it failed
 */
int FILE_PTRS::increment_seqno(){
    this->file_info.seqno += 1;
    return 1;
}


static char config_location[STRBUFFSIZE];

static char run_directory[STRBUFFSIZE];

static long long max_file_size = 0; //IN UNITS OF BYTES



static FILE_PTRS *data_files[MODULEINDEXSIZE] = {NULL};


/**
 * Create a file pointers for a given dome and module.
 * @param diskDir directory of the file created for the file pointer structure
 * @param dome dome number of the files
 * @param module module number of the files
 */
FILE_PTRS *data_file_init(const char *diskDir, int dome, int module) {
    time_t t = time(NULL);

    FILENAME_INFO filenameInfo(t, DP_STATIC_META, 0, dome, module, 0);
    return new FILE_PTRS(diskDir, &filenameInfo, "w");
}

/**
 * Write image header to file.
 * @param fileToWrite File the image header will be written to
 * @param dataHeader The output block header containing the image headers
 * @param blockIndex The block index for the specified output block
 */
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

/**
 * Writing the image module structure to file
 * @param dataBlock Data block of the containing the images to be written to disk
 * @param blockIndex The block index for the specified output block.
 */
int write_module_img_file(HSD_output_block_t *dataBlock, int blockIndex){
    FILE *fileToWrite;
    FILE_PTRS *moduleToWrite = data_files[dataBlock->header.img_pkt_head[blockIndex].mod_num];
    int mode = dataBlock->header.img_pkt_head[blockIndex].mode;
    int modSizeMultiplier = mode/8;

    if (mode == 16) {
        fileToWrite = moduleToWrite->bit16Img;
    } else if (mode == 8){
        fileToWrite = moduleToWrite->bit8Img;
    } else {
        printf("Mode %i not recognized\n", mode);
        printf("Module Header Value\n%s\n", dataBlock->header.img_pkt_head[blockIndex].toString().c_str());
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
        dataBlock->img_block + (blockIndex*MODULEDATASIZE));

    if (ftell(fileToWrite) > max_file_size){
        moduleToWrite->increment_seqno();
        if (mode == 16){
            moduleToWrite->new_dp_file(DP_BIT16_IMG, run_directory, "w");
        } else if (mode == 8){
            moduleToWrite->new_dp_file(DP_BIT8_IMG, run_directory, "w");
        }
    }

    return 1;
}

/**
 * Write the coincidence header information to file.
 * @param fileToWrite File the image header will be written to
 * @param dataHeader The output block header containing the image headers
 * @param blockIndex The block index for the specified output block
 */
int write_coinc_header_file(FILE *fileToWrite, HSD_output_block_header_t *dataHeader, int blockIndex){
    fprintf(fileToWrite,
    "{ acq_mode: %u, mod_num: %u, qua_num: %u, pkt_num : %u, pkt_nsec : %u, tv_sec : %li, tv_usec : %li}",
    dataHeader->coin_pkt_head[blockIndex].acq_mode,
    dataHeader->coin_pkt_head[blockIndex].mod_num,
    dataHeader->coin_pkt_head[blockIndex].qua_num,
    dataHeader->coin_pkt_head[blockIndex].pkt_num,
    dataHeader->coin_pkt_head[blockIndex].pkt_nsec,
    dataHeader->coin_pkt_head[blockIndex].tv_sec,
    dataHeader->coin_pkt_head[blockIndex].tv_usec
    );
}

/**
 * Writing the coincidence(Pulse Height) image to file
 * @param dataBlock Data block of the containing the images to be written to disk
 * @param blockIndex The block index for the specified output block.
 */
int write_module_coinc_file(HSD_output_block_t *dataBlock, int blockIndex){
    FILE *fileToWrite;
    FILE_PTRS *moduleToWrite = data_files[dataBlock->header.coin_pkt_head[blockIndex].mod_num];
    char mode = dataBlock->header.coin_pkt_head[blockIndex].acq_mode;

    if (mode == 0x1) {
        fileToWrite = moduleToWrite->PHImg;
    } else {
        printf("Mode %c not recognized\n", mode);
        printf("Module Header Value\n%s\n", dataBlock->header.img_pkt_head[blockIndex].toString().c_str());
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

    write_coinc_header_file(fileToWrite, &(dataBlock->header), blockIndex);

    pff_end_json(fileToWrite);

    pff_write_image(fileToWrite, 
        SCIDATASIZE*2, 
        dataBlock->coinc_block + (blockIndex*PKTDATASIZE));

    if (ftell(fileToWrite) > max_file_size){
        if (mode == 0x1){
            moduleToWrite->new_dp_file(DP_PH_IMG, run_directory, "w");
        }
    }
    return 1;
}

/**
 * Create data files from the provided config file.
 */
int create_data_files_from_config(){
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
                    data_files[modNum] = data_file_init(run_directory, 0, modNum);
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


typedef enum {
    DIR_EXISTS,
    DIR_DNE,
    NOT_DIR,
    DIR_READ_ERROR
} DIR_STATUS;

DIR_STATUS check_directory(char *run_directory){
    if (strlen(run_directory) <= 0){return DIR_DNE;}

    struct stat s;
    int err = stat(run_directory, &s);
    if (err == -1){
        if (ENOENT == errno){
            return DIR_DNE;
        } else {
            return DIR_READ_ERROR; 
        }
    } else {
        if (S_ISDIR(s.st_mode)) {
            return DIR_EXISTS;
        } else {
            return NOT_DIR;
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
    // Fetch user input for save location of data files.
    hgets(st.buf, "RUNDIR", STRBUFFSIZE, run_directory);
    // Remove old run directory so that info isn't saved for next run.
    hdel(st.buf, "RUNDIR");
    // Check to see if the run directory provided is an accurage directory
    switch(check_directory(run_directory)) {
        case DIR_EXISTS:
            printf("Run directory: %s\n", run_directory);
            break;
        case DIR_DNE:
            fprintf(stderr, "Directory %s does not exist\n", run_directory);
            exit(1);
        case NOT_DIR:
            fprintf(stderr, "%s is not a directory\n", run_directory);
            exit(1);
        case DIR_READ_ERROR:
            fprintf(stderr, "Issue reading directory %s\n", run_directory);
            exit(1);
            
    }
    //If directory doesn't end in a / then add it to the run_directory variable
    if (run_directory[strlen(run_directory) - 1] != '/') {
        char endingSlash = '/';
        strncat(run_directory, &endingSlash, 1);
    }

    // Fetch user input for config file location.
    sprintf(config_location, CONFIGFILE_DEFAULT);
    hgets(st.buf, "CONFIG", STRBUFFSIZE, config_location);
    printf("Config Location: %s\n", config_location);

    // Fetch user input for max file size of data files.
    int maxFileSizeInput;
    hgeti4(st.buf, "MAXFILESIZE", &maxFileSizeInput);
    max_file_size = maxFileSizeInput*1E6;
    printf("Max file size is %i megabytes\n", maxFileSizeInput);



    printf("\n---------------SETTING UP DATA File------------------\n");
    
    //Create data files based on given config file.
    create_data_files_from_config();

    printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n");
    printf("-----------Finished Setup of Output Thread-----------\n\n");    

    return 0;
}

/**
 * Close all allocated resources
 */
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

    printf("---------------Running Output Thread-----------------\n\n");

    /*Initialization of HASHPIPE Values*/
    // Local aliases to shorten access to args fields
    // Our input buffer happens to be a HSD_ouput_databuf
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

        
        for (int i = 0; i < db->block[block_idx].header.img_block_size; i++){
            write_module_img_file(&(db->block[block_idx]), i);
        }

        for (int i = 0; i < db->block[block_idx].header.coinc_block_size; i++){
            write_module_coinc_file(&(db->block[block_idx]), i);
        }

        if (QUITSIG) {
            printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n\n");
            QUITSIG = 0;
        }

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

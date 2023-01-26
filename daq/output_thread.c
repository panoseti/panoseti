// output_thread.c
//
// write data from output buffer to files

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
#include "databuf.h"
#include "pff.h"
#include "dp.h"

// Use this stdio buffer size (1M) for image-mode files.
// Default is 4K
//
#define IM_BUFSIZE 1048576

void increase_buffer(FILE* f, int bufsize) {
    char* b = (char*)malloc(bufsize);
    setvbuf(f, b, _IOFBF, bufsize);
}

// Structure for storing file pointers opened by output thread.
// A file is created for all possible data products described by pff.h
//
struct FILE_PTRS{
    DIRNAME_INFO dir_info;
    FILENAME_INFO file_info;
    int image_seqno, ph_seqno;
    FILE *bit16Img, *bit8Img, *PH256Img, *PH1024Img;
    FILE_PTRS(const char *diskDir, FILENAME_INFO *fi);
    void make_files(const char *diskDir);
    void new_dp_file(DATA_PRODUCT dp, const char *diskDir);
};

// Constructor for file pointer structure
// diskDir: directory used for writing all files monitored by file pointer
// fileInfo: file information structure stored by file pointer

FILE_PTRS::FILE_PTRS(const char *diskDir, FILENAME_INFO *fi){
    image_seqno = 0;
    ph_seqno = 0;
    file_info = *fi;
    make_files(diskDir);
}

static bool path_exists(const char* path) {
    struct stat buf;
    if (stat(path, &buf)) return false;
    return true;
}

// Create files for the file pointer stucture given a directory.
// diskDir: directory where the files will be created

void FILE_PTRS::make_files(const char *run_dir){
    char buf[256];
    string filename;
    
    sprintf(buf, "module_%d", file_info.module);
    if (!path_exists(buf)) {
        if (mkdir(buf, 0777)) {
            printf("Can't mkdir %s\n", buf);
            exit(0);
        }
    }
    file_info.seqno = 0;
    for (int dp = DP_BIT16_IMG; dp < DP_NONE; dp++){
        file_info.data_product = (DATA_PRODUCT)dp;
        file_info.bytes_per_pixel = bytes_per_pixel((DATA_PRODUCT)dp);
        file_info.make_filename(filename);
        sprintf(buf, "module_%d/%s/%s",
            file_info.module, run_dir, filename.c_str()
        );
        FILE *f = fopen(buf, "w");
        if (!f) {
            printf("Error: can't open file %s\n", buf);
            exit(0);
        }
        switch (dp){
            case DP_BIT16_IMG:
                increase_buffer(f, IM_BUFSIZE);
                bit16Img = f;
                break;
            case DP_BIT8_IMG:
                increase_buffer(f, IM_BUFSIZE);
                bit8Img = f;
                break;
            case DP_PH_256_IMG:
                PH256Img = f;
                break;
            case DP_PH_1024_IMG:
                increase_buffer(f, IM_BUFSIZE);
                PH1024Img = f;
                break;
            default:
                break;
        }
        printf("Created file %s\n", buf);
    }
}

// Create a new file for a specified data product within file structure.
// called when a certain data product file has reached max file size.
// dp: Data product of the file that needs to be created.
// diskDir: directory

void FILE_PTRS::new_dp_file(DATA_PRODUCT dp, const char *run_dir){
    string filename;
    char buf[256];

    file_info.seqno = (dp==DP_PH_256_IMG||dp==DP_PH_1024_IMG)?ph_seqno:image_seqno;
    file_info.data_product = (DATA_PRODUCT)dp;
    file_info.start_time = time(NULL);
    file_info.bytes_per_pixel = bytes_per_pixel(dp);
    file_info.make_filename(filename);
    sprintf(buf, "module_%d/%s/%s",
        file_info.module, run_dir, filename.c_str()
    );
    FILE* f = fopen(buf, "w");
    if (!f) {
        printf("Error: can't open file %s\n", buf);
        exit(0);
    }
    switch (dp){
        case DP_BIT16_IMG:
            increase_buffer(f, IM_BUFSIZE);
            fclose(bit16Img);
            bit16Img = f;
            break;
        case DP_BIT8_IMG:
            increase_buffer(f, IM_BUFSIZE);
            fclose(bit8Img);
            bit8Img = f;
            break;
        case DP_PH_256_IMG:
            fclose(PH256Img);
            PH256Img = f;
            break;
        case DP_PH_1024_IMG:
            increase_buffer(f, IM_BUFSIZE);
            fclose(PH1024Img);
            PH1024Img = f;
            break;
        default:
            break;
    }
    printf("new_dp_file(): created file %s\n", buf);
}

static char config_location[STR_BUFFER_SIZE];

static char run_directory[STR_BUFFER_SIZE];

static long long max_file_size = 0; //IN UNITS OF BYTES

static FILE_PTRS *data_files[MAX_MODULE_INDEX] = {NULL};


// Create a file pointers for a given module.
// diskDir: directory of the file created for the file pointer structure
// module: module number of the files

FILE_PTRS* data_file_init(const char *diskDir, int module) {
    time_t t = time(NULL);

    FILENAME_INFO fi(t, DP_NONE, 0, module, 0);
    return new FILE_PTRS(diskDir, &fi);
}

// Write image header as JSON
// TOTAL SIZE MUST BE FIXED (but it doesn't matter what the size is)
//
int write_img_header_json(
    FILE *f, HSD_output_block_header_t *dataHeader, int frameIndex
){
    fprintf(f, "{\n");
    for (int i=0; i<QUABO_PER_MODULE; i++){
        fprintf(f,
        "   \"quabo_%1u\": { \"pkt_num\": %10u, \"pkt_tai\": %4u, \"pkt_nsec\": %9u, \"tv_sec\": %10li, \"tv_usec\": %6li}",
        i,
        dataHeader->img_mod_head[frameIndex].pkt_head[i].pkt_num,
        dataHeader->img_mod_head[frameIndex].pkt_head[i].pkt_tai,
        dataHeader->img_mod_head[frameIndex].pkt_head[i].pkt_nsec,
        dataHeader->img_mod_head[frameIndex].pkt_head[i].tv_sec,
        dataHeader->img_mod_head[frameIndex].pkt_head[i].tv_usec
        );
        if (i < QUABO_PER_MODULE-1){
            fprintf(f, ", ");
        }
        fprintf(f, "\n");
    }
    fprintf(f, "}");
    return 0;
}

// Write the image module structure to file
//
int write_module_img_file(HSD_output_block_t *dataBlock, int frameIndex){
    FILE *f;
    FILE_PTRS *moduleToWrite = data_files[dataBlock->header.img_mod_head[frameIndex].mod_num];
    int bits_per_pixel = dataBlock->header.img_mod_head[frameIndex].bits_per_pixel;
    int modSizeMultiplier = bits_per_pixel/8;

    if (bits_per_pixel == 16) {
        f = moduleToWrite->bit16Img;
    } else if (bits_per_pixel == 8){
        f = moduleToWrite->bit8Img;
    } else {
        printf("BPP %i not recognized\n", bits_per_pixel);
        printf("Module Header Value\n%s\n", dataBlock->header.img_mod_head[frameIndex].toString().c_str());
        return 0;
    }
    
    if (moduleToWrite == NULL){
        printf("Module To Write is null\n");
        return 0;
    } else if (f == NULL){
        printf("File to Write is null\n");
        return 0;
    } 

    pff_start_json(f);

    write_img_header_json(f, &(dataBlock->header), frameIndex);

    pff_end_json(f);

    pff_write_image(f, 
        QUABO_PER_MODULE*PIXELS_PER_IMAGE*modSizeMultiplier, 
        dataBlock->img_block + (frameIndex*BYTES_PER_MODULE_FRAME)
    );

    if (max_file_size && (ftell(f) > max_file_size)){
        moduleToWrite->image_seqno++;
        if (bits_per_pixel == 16){
            moduleToWrite->new_dp_file(DP_BIT16_IMG, run_directory);
        } else if (bits_per_pixel == 8){
            moduleToWrite->new_dp_file(DP_BIT8_IMG, run_directory);
        }
    }

    return 1;
}

// Write PH header information as JSON.  Fixed-length format.
// Note:
//      - If hashpipe is in grouping mode (group_ph_frames = 1), writes all 4 packet headers stored in the PH image header at frameIndex.
//      - Otherwise (group_ph_frames = 0), writes only the packet header at index 0 in the PH image header at frameIndex.
//
int write_ph_header_json(
    FILE *f, HSD_output_block_header_t *dataHeader, int frameIndex
){
    if (dataHeader->ph_img_head[frameIndex].group_ph_frames) {
        // Frame grouping is enabled. Write a 1024 pixel PH image.
        //
        fprintf(f, "{\n");
        for (int i=0; i<QUABO_PER_MODULE; i++){
            fprintf(f,
            "   \"quabo_%1u\": { \"pkt_num\": %10u, \"pkt_tai\": %4u, \"pkt_nsec\": %9u, \"tv_sec\": %10li, \"tv_usec\": %6li}",
            i,
            dataHeader->ph_img_head[frameIndex].pkt_head[i].pkt_num,
            dataHeader->ph_img_head[frameIndex].pkt_head[i].pkt_tai,
            dataHeader->ph_img_head[frameIndex].pkt_head[i].pkt_nsec,
            dataHeader->ph_img_head[frameIndex].pkt_head[i].tv_sec,
            dataHeader->ph_img_head[frameIndex].pkt_head[i].tv_usec
            );
            if (i < QUABO_PER_MODULE-1){
                fprintf(f, ", ");
            }
            fprintf(f, "\n");
        }
        fprintf(f, "}");
    } else {
        fprintf(f,
            "{ \"quabo_num\": %1u, \"pkt_num\": %10u, \"pkt_tai\": %4u, \"pkt_nsec\": %9u, \"tv_sec\": %10li, \"tv_usec\": %6li}",
            dataHeader->ph_img_head[frameIndex].pkt_head[0].quabo_num,
            dataHeader->ph_img_head[frameIndex].pkt_head[0].pkt_num,
            dataHeader->ph_img_head[frameIndex].pkt_head[0].pkt_tai,
            dataHeader->ph_img_head[frameIndex].pkt_head[0].pkt_nsec,
            dataHeader->ph_img_head[frameIndex].pkt_head[0].tv_sec,
            dataHeader->ph_img_head[frameIndex].pkt_head[0].tv_usec
        );
    }
    return 0;
}

// Write a Pulse Height image to file
// dataBlock: Data block of the images to be written
// frameIndex: The frame index for the specified output block.
// Note: 
//      - If hashpipe is in grouping mode (group_ph_frames = 1), writes the entire PH image block at frameIndex.
//      - Otherwise (group_ph_frames = 0), writes only the first 512 bytes of the PH image block at frameIndex.
//
int write_module_ph_file(HSD_output_block_t *dataBlock, int frameIndex){
    FILE *f;
    FILE_PTRS *moduleToWrite = data_files[dataBlock->header.ph_img_head[frameIndex].mod_num];
    int group_ph_frames = dataBlock->header.ph_img_head[frameIndex].group_ph_frames;
    int num_ph_frames_to_write;

    if (group_ph_frames) {
        f = moduleToWrite->PH1024Img;
        num_ph_frames_to_write = 4;
    } else {
        f = moduleToWrite->PH256Img;
        num_ph_frames_to_write = 1;
    }

    if (moduleToWrite == NULL){
        printf("Module To Write is null\n");
        return 0;
    } else if (f == NULL){
        printf("File to Write is null\n");
        return 0;
    } 

    pff_start_json(f);

    write_ph_header_json(f, &(dataBlock->header), frameIndex);

    pff_end_json(f);

    // NOTE: when group_ph_frames=0, only the first 512 bytes of the PH data block
    // will contain meaningful data.
    pff_write_image(f, 
        num_ph_frames_to_write*PIXELS_PER_IMAGE*2, 
        dataBlock->ph_block + (frameIndex*BYTES_PER_PH_FRAME)
    );

    if (max_file_size && (ftell(f) > max_file_size)){
        moduleToWrite->ph_seqno++;
        if (mode == 0x1){
            if (group_ph_frames) {
                moduleToWrite->new_dp_file(DP_PH_1024_IMG, run_directory);
            } else {
                moduleToWrite->new_dp_file(DP_PH_256_IMG, run_directory);
            }
        }
    }
    return 1;
}

// Create data files from the provided config file.

int create_data_files_from_config(){
    FILE *configFile = fopen(config_location, "r");
    char fbuf[STR_BUFFER_SIZE];
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
                    data_files[modNum] = data_file_init(run_directory, modNum);
                    printf("Created Data file for Module %u\n", modNum);
                }
            }
        } else {
            if (fgets(fbuf, STR_BUFFER_SIZE, configFile) == NULL) {
                break;
            }
        }
        cbuf = getc(configFile);
    }

    if (fclose(configFile) == EOF) {
        printf("Warning: Unable to close module configuration file.\n");
    }
    return 0;
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


// Signal handler to allow for hashpipe to exit gracfully
// and to allow for creating of new files by command.
//
static int QUITSIG;

void QUIThandler(int signum) {
    QUITSIG = 1;
}

static int init(hashpipe_thread_args_t *args) {
    // Get info from status buffer if present
    hashpipe_status_t st = args->st;
    printf("\n\n-----------Start Setup of Output Thread--------------\n");

    // Fetch user input for save location of data files.
    hgets(st.buf, "RUNDIR", STR_BUFFER_SIZE, run_directory);

    // Remove old run directory so that info isn't saved for next run.
    hdel(st.buf, "RUNDIR");

    // Check if the run directory exits
    //
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
    // If directory doesn't end in a / then add it to the run_directory variable
    if (run_directory[strlen(run_directory) - 1] != '/') {
        char endingSlash = '/';
        strncat(run_directory, &endingSlash, 1);
    }

    // Fetch user input for config file location.
    sprintf(config_location, CONFIGFILE_DEFAULT);
    hgets(st.buf, "CONFIG", STR_BUFFER_SIZE, config_location);
    printf("Config Location: %s\n", config_location);

    // Fetch user input for max file size of data files.
    int maxFileSizeInput;
    hgeti4(st.buf, "MAXFILESIZE", &maxFileSizeInput);
    max_file_size = maxFileSizeInput*1E6;
    printf("Max file size is %i megabytes\n", maxFileSizeInput);

    printf("\n---------------SETTING UP DATA File------------------\n");
    
    // Create data files based on given config file.
    create_data_files_from_config();

    printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n");
    printf("-----------Finished Setup of Output Thread-----------\n\n");    

    return 0;
}

void close_files() {
    for (int i = 0; i < MAX_MODULE_INDEX; i++){
        if (data_files[i] != NULL){
            fclose(data_files[i]->bit16Img);
            fclose(data_files[i]->bit8Img);
            fclose(data_files[i]->PH256Img);
            fclose(data_files[i]->PH1024Img);
        }
    }
}

static void *run(hashpipe_thread_args_t *args) {
    signal(SIGQUIT, QUIThandler);
    QUITSIG = 0;

    printf("---------------Running Output Thread-----------------\n\n");

    // Initialization of HASHPIPE Values
    // Local aliases to shorten access to args fields
    // Our input buffer happens to be a HSD_ouput_databuf
    //
    HSD_output_databuf_t *db = (HSD_output_databuf_t *)args->ibuf;
    hashpipe_status_t st = args->st;
    const char *status_key = args->thread_desc->skey;

    int rv;
    int block_idx = 0;
    uint64_t mcnt = 0;
    FILE_PTRS *currentDataFile;

    // Main loop
    while (run_threads()) {

        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "OUTBLKIN", block_idx);
        hputi8(st.buf, "OUTMCNT", mcnt);
        hputs(st.buf, status_key, "waiting");
        hashpipe_status_unlock_safe(&st);

        // Wait for the output buffer to be free
        while ((rv = HSD_output_databuf_wait_filled(db, block_idx)) != HASHPIPE_OK) {
            if (rv == HASHPIPE_TIMEOUT) {
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

        
        for (int i = 0; i < db->block[block_idx].header.n_img_module; i++){
            write_module_img_file(&(db->block[block_idx]), i);
        }

        for (int i = 0; i < db->block[block_idx].header.n_ph_img; i++){
            write_module_ph_file(&(db->block[block_idx]), i);
        }

        if (QUITSIG) {
            printf("Use Ctrl+\\ to create a new file and Ctrl+c to close program\n\n");
            QUITSIG = 0;
        }

        if (db->block[block_idx].header.INTSIG) {
            close_files();
            printf("OUTPUT_THREAD Ended\n");
            break;
        }

        HSD_output_databuf_set_free(db, block_idx);
        block_idx = (block_idx + 1) % db->header.n_block;
        mcnt++;

        // exit if thread has been cancelled
        pthread_testcancel();
    }

    printf("Returned Output_thread\n");
    return THREAD_OK;
}

// Sets the functions and buffers for this thread

static hashpipe_thread_desc_t HSD_output_thread = {
    name : "output_thread",
    skey : "OUTSTAT",
    init : init,
    run : run,
    ibuf_desc : {HSD_output_databuf_create},
    obuf_desc : {NULL}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&HSD_output_thread);
}

/* compute_thread.c
 *
 * Does pre processing on the data coming from the quabos before writing to file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

#include "hashpipe.h"

#include "databuf.h"
#include "compute_thread.h"
#include "image.h"
#include "process_frame.h"

// Number of mode and also used the create the size of array (Modes 1,2,3,6,7)
#define NUM_OF_MODES 7

/**
 * Writes the image data from the module data into block in output buffer.
 * @param mod_data Module data passed in be copied to output block.
 * @param out_block Output block to be written to.
 */
void write_frame_to_out_buffer(
    module_data_t* mod_data, HSD_output_block_t* out_block
){
    int out_index = out_block->header.n_img_module;
    HSD_output_block_header_t* out_header = &(out_block->header);
    
    mod_data->mod_head.copy_to(&(out_header->img_mod_head[out_index]));
    
    memcpy(
        out_block->img_block + (out_index * BYTES_PER_MODULE_FRAME),
        mod_data->data,
        sizeof(uint8_t)*BYTES_PER_MODULE_FRAME
    );

    out_block->header.n_img_module++;
}

/**
 * Write coincidence(Pulse Height) images into the output buffer.
 * @param in_block Input data block containing the image needed to be copied.
 * @param pktIndex Packet index for the image in the input data block.
 * @param out_block Output data block to be written to. 
 */
void write_coinc_to_out_buffer(
    HSD_input_block_t* in_block,
    int pktIndex,
    HSD_output_block_t* out_block
) {
    int out_index = out_block->header.n_coinc_img;

    in_block->header.pkt_head[pktIndex].copy_to(&(out_block->header.coinc_pkt_head[out_index]));
    
    memcpy(
        out_block->coinc_block + out_index*BYTES_PER_PKT_IMAGE,
        in_block->data_block + pktIndex*BYTES_PER_PKT_IMAGE,
        sizeof(in_block->data_block[0])*BYTES_PER_PKT_IMAGE
    );

    out_block->header.n_coinc_img++;
}

// copy quabo image to module image buffer
// If appropriate, copy module image to output buffer first
//
void storeData(
    module_data_t* mod_data,        // module image
    HSD_input_block_t* in_block,    // block in input buffer (quabo images)
    HSD_output_block_t* out_block,  // block in output buffer
    int pktIndex                    // index in input buffer
){
    int mode, bytes_per_pixel;
    packet_header_t *pkt_head = &(in_block->header.pkt_head[pktIndex]);
    uint32_t nanosec = pkt_head->pkt_nsec;
    int quabo_num = pkt_head->qua_num;

    uint8_t currentStatus = (0x01 << quabo_num);

    // see what kind of packet it is

    if (pkt_head->acq_mode == 0x1){
        //PH Mode
        write_coinc_to_out_buffer(in_block, pktIndex, out_block);
        return;
    } else if(pkt_head->acq_mode == 0x2 || pkt_head->acq_mode == 0x3){
        //16 bit Imaging mode
        bytes_per_pixel = 2;
        mode = 16;
    } else if (pkt_head->acq_mode == 0x6 || pkt_head->acq_mode == 0x7){
        //8 bit Imaging mode
        bytes_per_pixel = 1;
        mode = 8;
    } else {
        fprintf(stderr, "Unknown acqmode %X\n ", pkt_head->acq_mode);
        fprintf(stderr, "moduleNum=%X quaboNum=%X PKTNUM=%X\n",
            pkt_head->mod_num, quabo_num, pkt_head->pkt_num
        );
        fprintf(stderr, "packet skipped\n");
        return;
    }

    // set min/max times of quabo images in module image
    //
    if(mod_data->status == 0){
        // Empty module pair obj
        // set both the upper and lower limit to current time
        mod_data->mod_head.mod_num = in_block->header.pkt_head[pktIndex].mod_num;
        mod_data->mod_head.mode = mode;
        mod_data->max_nanosec = nanosec;
        mod_data->min_nanosec = nanosec;
    } else if(nanosec > mod_data->max_nanosec){
        mod_data->max_nanosec = nanosec;
    } else if (nanosec < mod_data->min_nanosec){
        mod_data->min_nanosec = nanosec;
    }

    // see if we should add module frame to output buffer
    // - the quabo position of the new packet is already filled in module buf
    // - or mode is different (???)
    // - or time threshold is exceeded
    //
    if ((mod_data->status & currentStatus) 
        || mod_data->mod_head.mode != mode 
        || (mod_data->max_nanosec - mod_data->min_nanosec) > NANOSEC_THRESHOLD
    ) {
        
        // A module frame is now final.
        // do long pulse finding or other stuff here.
        //
        //process_frame(mod_data);

        write_frame_to_out_buffer(mod_data, out_block);

        // clear module frame buffer
        mod_data->clear();
    }

#if 1
    // copy rotated quabo image to module image
    //
    if (bytes_per_pixel == 1) {
        void *p = in_block->data_block + (pktIndex*BYTES_PER_PKT_IMAGE);
        quabo8_to_module8_copy(
            (QUABO_IMG8&)p,
            quabo_num,
            (MODULE_IMG8&)(mod_data->data)
        );
    } else {
        void *p = in_block->data_block + (pktIndex*BYTES_PER_PKT_IMAGE);
        quabo16_to_module16_copy(
            (QUABO_IMG16&)p,
            quabo_num,
            (MODULE_IMG16&)(mod_data->data)
        );
    }
#else
    memcpy(
        mod_data->data + (quabo_num*PIXELS_PER_IMAGE*bytes_per_pixel),
        in_block->data_block + (pktIndex*BYTES_PER_PKT_IMAGE),
        sizeof(uint8_t)*PIXELS_PER_IMAGE*bytes_per_pixel
    );
#endif
    
    // copy the header
    //
    in_block->header.pkt_head[pktIndex].copy_to(
        &(mod_data->mod_head.pkt_head[quabo_num])
    );

    //Mark the status for the packet slot as taken
    mod_data->status = mod_data->status | currentStatus;
    mod_data->mod_head.mod_num = in_block->header.pkt_head[pktIndex].mod_num;
    mod_data->mod_head.mode = mode;
}


/**
 * Structure of quabo stored for determining packet loss
 */
typedef struct quabo_info{
    uint16_t prev_pkt_num[NUM_OF_MODES+1];
    int lost_pkts[NUM_OF_MODES+1];
} quabo_info_t;

/**
 * Initializing a new quabo_info object
 */
quabo_info_t* quabo_info_t_new(){
    quabo_info_t* value = (quabo_info_t*) malloc(sizeof(struct quabo_info));
    memset(value->lost_pkts, -1, sizeof(value->lost_pkts));
    memset(value->prev_pkt_num, 0, sizeof(value->prev_pkt_num));
    return value;
}

//A module index for holding the data structures for the modules it expects.
static module_data_t* moduleInd[MAX_MODULE_INDEX] = {NULL};

/**
 * Initialization function for Hashpipe. This function is called once when the thread is created
 * @param args Arugments passed in by hashpipe framework.
 */
static int init(hashpipe_thread_args_t * args){
    hashpipe_status_t st = args->st;
    //Initialize the INTSIG signal within the buffer to be zero
    printf("\n\n-----------Start Setup of Compute Thread--------------\n");
    HSD_output_databuf_t *db_out = (HSD_output_databuf_t *)args->obuf;
    for (int i = 0 ; i < db_out->header.n_block; i++){
        db_out->block[i].header.INTSIG = 0;
    }

    //Initializing the module data with the config file
    char config_location[STR_BUFFER_SIZE];
    sprintf(config_location, CONFIGFILE_DEFAULT);
    hgets(st.buf, "CONFIG", STR_BUFFER_SIZE, config_location);
    printf("Config Location: %s\n", config_location);
    FILE *modConfig_file = fopen(config_location, "r");

    char fbuf[100];
    char cbuf;
    unsigned int modName;

    if (modConfig_file == NULL) {
        perror("Error Opening Config File");
        exit(1);
    }
    cbuf = getc(modConfig_file);

    //Parsing the Module Config file for the modules to expect data from
    //Creates structures for holding that data in the module index
    while(cbuf != EOF){
        ungetc(cbuf, modConfig_file);
        if (cbuf != '#'){
            if (fscanf(modConfig_file, "%u\n", &modName) == 1){
                if (moduleInd[modName] == NULL){

                    moduleInd[modName] = new module_data();

                    fprintf(stdout, "Created Module: %u.%u-%u\n", 
                    (unsigned int) (modName << 2)/0x100, (modName << 2) % 0x100, ((modName << 2) % 0x100) + 3);
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
        fprintf(stderr, "Warning: Unable to close module configuration file.\n");
    }
    printf("-----------Finished Setup of Compute Thread-----------\n\n");
    
    return 0;

    
}

/**
 * Main run function that is ran once when the threads are running. To keep thread running
 * make sure to use a while loop.
 * @param args Arguements passed in by the hashpipe framework
 */
static void *run(hashpipe_thread_args_t * args){
    printf("\n---------------Running Compute Thread-----------------\n\n");
    // Local aliases to shorten access to args fields
    HSD_input_databuf_t *db_in = (HSD_input_databuf_t *)args->ibuf;
    HSD_output_databuf_t *db_out = (HSD_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    // Index values for the circular buffers in the shared buffer with the input and output threads
    int rv;
    uint64_t mcnt=0;
    int curblock_in=0;
    int curblock_out=0;
    int INTSIG;

    //Variables to display pkt info
    uint8_t mode;                                       //The current mode of the packet block
    quabo_info_t* quaboInd[0xffff] = {NULL};            //Create a rudimentary hash map of the quabo number and linked list ind

    quabo_info_t* currentQuabo;                         //Pointer to the quabo info that is currently being used
    uint16_t boardLoc;                                  //The boardLoc(quabo index) for the current packet
    char* boardLocstr = (char *)malloc(sizeof(char)*10);
    
    //Counters for the packets lost
    int total_lost_pkts = 0;
    int current_pkt_lost;

    
    while(run_threads()){
        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "COMBLKIN", curblock_in);
        hputs(st.buf, status_key, "waiting");
        hputi4(st.buf, "COMBKOUT", curblock_out);
	    hputi8(st.buf,"COMMCNT",mcnt);
        hashpipe_status_unlock_safe(&st);

        //Wait for new input block to be filled
        while ((rv=HSD_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
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

        // Wait for new output block to be free
        while ((rv=HSD_output_databuf_wait_free(db_out, curblock_out)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked compute out");
                hashpipe_status_unlock_safe(&st);
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                pthread_exit(NULL);
                break;
            }
        }

        //Note processing status
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "processing packet");
        hashpipe_status_unlock_safe(&st);

        //Resetting the values in the new output block
        db_out->block[curblock_out].header.n_img_module = 0;
        db_out->block[curblock_out].header.n_coinc_img = 0;
        db_out->block[curblock_out].header.INTSIG = db_in->block[curblock_in].header.INTSIG;
        INTSIG = db_in->block[curblock_in].header.INTSIG;

        uint16_t moduleNum;
        for(int i = 0; i < db_in->block[curblock_in].header.n_pkts_in_block; i++){
            //----------------CALCULATION BLOCK-----------------
            moduleNum = db_in->block[curblock_in].header.pkt_head[i].mod_num;

            if (moduleInd[moduleNum] == NULL){
                fprintf(stderr, "Detected New Module not in Config File: %u.%u\n", (unsigned int) (moduleNum << 2)/0x100, (moduleNum << 2) % 0x100);
                fprintf(stderr, "Packet skipping\n");
                continue;
            }

            storeData(moduleInd[moduleNum], &(db_in->block[curblock_in]), &(db_out->block[curblock_out]), i);
            
            //------------End CALCULATION BLOCK----------------


            //Finding the packet number and computing the lost of packets by using packet number
            //Read the packet number from the packet
            mode = db_in->block[curblock_in].header.pkt_head[i].acq_mode;
            boardLoc = db_in->block[curblock_in].header.pkt_head[i].mod_num * 4 + db_in->block[curblock_in].header.pkt_head[i].qua_num;

            //Check to see if there is a quabo info for the current quabo packet. If not create an object
            if (quaboInd[boardLoc] == NULL){
                quaboInd[boardLoc] = quabo_info_t_new();            //Create a new quabo info object

                printf("New Quabo Detected ID:%u.%u\n", (boardLoc >> 8) & 0x00ff, boardLoc & 0x00ff); //Output the terminal the new quabo
            }

            //Set the current Quabo to the one stored in memory
            currentQuabo = quaboInd[boardLoc];

            //Check to see if it is newly created quabo info if so then inialize the lost packet number to 0
            if (currentQuabo->lost_pkts[mode] < 0) {
                currentQuabo->lost_pkts[mode] = 0;
            } else {
                //Check to see if the current packet number is less than the previous. If so the number has overflowed and looped.
                //Compenstate for this if this has happend, and then take the difference of the packet numbers minus 1 to be the packets lost
                if (db_in->block[curblock_in].header.pkt_head[i].pkt_num < currentQuabo->prev_pkt_num[mode])
                    current_pkt_lost = (0xffff - currentQuabo->prev_pkt_num[mode]) + db_in->block[curblock_in].header.pkt_head[i].pkt_num;
                else
                    current_pkt_lost = (db_in->block[curblock_in].header.pkt_head[i].pkt_num - currentQuabo->prev_pkt_num[mode]) - 1;
                
                currentQuabo->lost_pkts[mode] += current_pkt_lost; //Add this packet lost to the total for this quabo
                total_lost_pkts += current_pkt_lost;               //Add this packet lost to the overall total for all quabos
            }
            currentQuabo->prev_pkt_num[mode] = db_in->block[curblock_in].header.pkt_head[i].pkt_num; //Update the previous packet number to be the current packet number
        }

        /*Update input and output block for both buffers*/
        //Mark output block as full and advance
        HSD_output_databuf_set_filled(db_out, curblock_out);
        curblock_out = (curblock_out + 1) % db_out->header.n_block;

        //Mark input block as free and advance
        HSD_input_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->header.n_block;
        mcnt++;

        //Break out when SIGINT is found
        if(INTSIG) {
            printf("COMPUTE_THREAD Ended\n");
            break;
        }

        sprintf(boardLocstr, "%u.%u", (boardLoc >> 8) & 0x00ff, boardLoc & 0x00ff);
        //display packetnum in status

        if (currentQuabo){
            hashpipe_status_lock_safe(&st);
            hputi4(st.buf, "TPKTLST", total_lost_pkts);
            hputi4(st.buf, "M1PKTLST", currentQuabo->lost_pkts[1]);
            hputi4(st.buf, "M2PKTLST", currentQuabo->lost_pkts[2]);
            hputi4(st.buf, "M3PKTLST", currentQuabo->lost_pkts[3]);
            hputi4(st.buf, "M6PKTLST", currentQuabo->lost_pkts[6]);
            hputi4(st.buf, "M7PKTLST", currentQuabo->lost_pkts[7]);
            hashpipe_status_unlock_safe(&st);
        }

        //Check for cancel
        pthread_testcancel();
    }

    printf("Returned Compute_thread\n");
    return THREAD_OK;
}

/**
 * Sets the functions and buffers for this thread
 */
static hashpipe_thread_desc_t HSD_compute_thread = {
    name: "compute_thread",
    skey: "COMPUTESTAT",
    init: init,
    run: run,
    ibuf_desc: {HSD_input_databuf_create},
    obuf_desc: {HSD_output_databuf_create}
};

static __attribute__((constructor)) void ctor(){
    register_hashpipe_thread(&HSD_compute_thread);
}

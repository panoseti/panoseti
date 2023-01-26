// compute_thread.c
// get data from input buffer, combine/transform it,
// write results to output buffer

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

// Number of DAQ modes
//
#define NUM_OF_MODES 7

// Store user input for the GROUPFRAMES option. 
// Equal to 0 (write 256 pixel PH images) or 1 (write 1024 pixel PH images).
//
static int group_ph_frames;

// copy image data from the module data to output buffer.
//
void write_frame_to_out_buffer(
    MODULE_IMAGE_BUFFER* mod_data,
    HSD_output_block_t* out_block
) {
    int out_index = out_block->header.n_img_module;
    HSD_output_block_header_t* out_header = &(out_block->header);
    
    out_header->img_mod_head[out_index] = mod_data->mod_head;
    
    memcpy(
        out_block->img_block + (out_index * BYTES_PER_MODULE_FRAME),
        mod_data->data,
        BYTES_PER_MODULE_FRAME
    );

    out_block->header.n_img_module++;
}

// Copy a pulse height image to the output buffer.
//
void write_ph_to_out_buffer(
    PH_IMAGE_BUFFER* ph_data,
    HSD_output_block_t* out_block      // block in output buffer
) {
    int out_index = out_block->header.n_ph_img;
    HSD_output_block_header_t* out_header = &(out_block->header);
    
    out_header->ph_img_head[out_index] = ph_data->ph_head;
    
    memcpy(
        out_block->ph_block + (out_index * BYTES_PER_PH_FRAME),
        ph_data->data,
        BYTES_PER_PH_FRAME
    );
    out_block->header.n_ph_img++;
}


// copy quabo image to module image buffer
// If needed, copy module image to output buffer first
//
void storeData(
    MODULE_IMAGE_BUFFER* mod_data,        // module image
    PH_IMAGE_BUFFER* ph_data,             // PH 1024 image
    HSD_input_block_t* in_block,    // block in input buffer (quabo images)
    HSD_output_block_t* out_block,  // block in output buffer (module images)
    int pktIndex                    // index in input buffer
        // TODO: pass the packet header rather than the index
){
    int bits_per_pixel, bytes_per_pixel;
    PACKET_HEADER *pkt_head = &(in_block->header.pkt_head[pktIndex]);
    uint32_t nanosec = pkt_head->pkt_nsec;
    int quabo_num = pkt_head->quabo_num;

    uint8_t quabo_bit = 1 << quabo_num;

    // see what kind of packet it is
    bool is_ph_packet = false;
    if (pkt_head->acq_mode == 0x1){
        //PH Mode
        is_ph_packet = true;
    } else if(pkt_head->acq_mode == 0x2 || pkt_head->acq_mode == 0x3){
        //16 bit Imaging mode
        bytes_per_pixel = 2;
        bits_per_pixel = 16;
    } else if (pkt_head->acq_mode == 0x6 || pkt_head->acq_mode == 0x7){
        //8 bit Imaging mode
        bytes_per_pixel = 1;
        bits_per_pixel = 8;
    } else {
        fprintf(stderr, "Unknown acqmode %X\n ", pkt_head->acq_mode);
        fprintf(stderr, "moduleNum=%X quaboNum=%X PKTNUM=%X\n",
            pkt_head->mod_num, quabo_num, pkt_head->pkt_num
        );
        fprintf(stderr, "packet skipped\n");
        return;
    }

    if (!is_ph_packet) {
        //--------------Process packet as part of a module image--------------
        //

        // set min/max times of quabo images in module image
        //
        if (mod_data->quabos_bitmap == 0){
            // no quabo images yet.
            // set both the upper and lower limit to current time
            //
            mod_data->mod_head.mod_num = in_block->header.pkt_head[pktIndex].mod_num;
            mod_data->mod_head.bits_per_pixel = bits_per_pixel;
            mod_data->max_nanosec = nanosec;
            mod_data->min_nanosec = nanosec;
        } else if (nanosec > mod_data->max_nanosec){
            mod_data->max_nanosec = nanosec;
        } else if (nanosec < mod_data->min_nanosec){
            mod_data->min_nanosec = nanosec;
        }

        // see if we should add module frame to output buffer
        // - the quabo position of the new packet is already filled in module buf
        // - or bytes/pixel is different (should never happen)
        // - or time threshold is exceeded
        //
        bool do_write = false;
        if (mod_data->quabos_bitmap & quabo_bit) {
            //printf("bit already set: %d %d\n", mod_data->quabos_bitmap, quabo_bit);
            do_write = true;
        } else if (mod_data->mod_head.bits_per_pixel != bits_per_pixel) {
            //printf("new bits_per_pixel %d %d\n", mod_data->mod_head.bits_per_pixel, bits_per_pixel);
            do_write = true;
        } else if (mod_data->max_nanosec - mod_data->min_nanosec > IMG_NANOSEC_THRESHOLD) {
            //printf("elapsed time %d %d\n", mod_data->max_nanosec, mod_data->min_nanosec);
            do_write = true;
        }
        if (do_write) {    
            // A module frame is now final.
            // do long pulse finding or other stuff here.
            //
            // process_frame(mod_data);

            write_frame_to_out_buffer(mod_data, out_block);
            
            // clear module frame buffer
            mod_data->clear();
            mod_data->max_nanosec = nanosec;
            mod_data->min_nanosec = nanosec;
        }

        // copy rotated quabo image to module image
        //
        if (bytes_per_pixel == 1) {
            void *p = in_block->data_block + (pktIndex*BYTES_PER_PKT_IMAGE);
            quabo8_to_module8_copy(
                p,
                quabo_num,
                mod_data->data
            );
        } else {
            void *p = in_block->data_block + (pktIndex*BYTES_PER_PKT_IMAGE);
            quabo16_to_module16_copy(
                p,
                quabo_num,
                mod_data->data
            );
        }
        
        // copy the header
        //
        mod_data->mod_head.pkt_head[quabo_num] = in_block->header.pkt_head[pktIndex];

        // Mark the quabo slot as taken
        //
        mod_data->quabos_bitmap |= quabo_bit;
        mod_data->mod_head.mod_num = in_block->header.pkt_head[pktIndex].mod_num;
        mod_data->mod_head.bits_per_pixel = bits_per_pixel;
    } else {
        //--------------Process packet as a PH 256 image--------------
        //
        if (!group_ph_frames) {
            // Store header metadata
            ph_data->ph_head.mod_num = in_block->header.pkt_head[pktIndex].mod_num;
            ph_data->ph_head.group_ph_frames = group_ph_frames; 
            // copy packet header 
            // Note: when grouping is disabled, all headers are stored at
            // index 0 of the packet header array for this block
            //
            ph_data->ph_head.pkt_head[0] = in_block->header.pkt_head[pktIndex];
 
            // rotate the image and copy to first 512 bytes of the data array in ph_data.
            //
            void *p = in_block->data_block + (pktIndex*BYTES_PER_PKT_IMAGE);
            quabo16_to_quabo16_copy(
                p,
                quabo_num,
                ph_data->data
            );
            write_ph_to_out_buffer(ph_data, out_block);

            // clear PH frame buffer
            ph_data->clear();
            ph_data->max_nanosec = nanosec;
            ph_data->min_nanosec = nanosec;
            return;
        }
        //--------------Process packet as part of a PH 1024 image-------------- 
        //
        
        // set min/max times of quabo PH 256 pixel images in PH 1024 pixel image
        //
        if (ph_data->quabos_bitmap == 0){
            // no quabo images yet.
            // set both the upper and lower limit to current time
            //
            ph_data->ph_head.mod_num = in_block->header.pkt_head[pktIndex].mod_num;
            ph_data->ph_head.group_ph_frames = group_ph_frames;
            ph_data->max_nanosec = nanosec;
            ph_data->min_nanosec = nanosec;
        } else if (nanosec > ph_data->max_nanosec){
            ph_data->max_nanosec = nanosec;
        } else if (nanosec < ph_data->min_nanosec){
            ph_data->min_nanosec = nanosec;
        }

        // see if we should add PH 1024 frame to output buffer
        // - the quabo position of the new packet is already filled in pulse-height buf
        // - or bytes/pixel is different (should never happen)
        // - or time threshold is exceeded
        //
        bool do_write = false;
        if (ph_data->quabos_bitmap & quabo_bit) {
            //printf("bit already set: %d %d\n", mod_data->quabos_bitmap, quabo_bit);
            do_write = true;
        } else if (ph_data->max_nanosec - ph_data->min_nanosec > PH_NANOSEC_THRESHOLD) {
            //printf("elapsed time %d %d\n", mod_data->max_nanosec, mod_data->min_nanosec);
            do_write = true;
        }
        if (do_write) {    
            // A PH 1024 frame is now final.

            write_ph_to_out_buffer(ph_data, out_block);

            // clear PH frame buffer
            ph_data->clear();
            ph_data->max_nanosec = nanosec;
            ph_data->min_nanosec = nanosec;
        }

        // copy rotated quabo image to PH 1024 image
        //
        void *p = in_block->data_block + (pktIndex*BYTES_PER_PKT_IMAGE);
        quabo16_to_module16_copy(
            p,
            quabo_num,
            ph_data->data
        );
        
        // copy the header
        //
        ph_data->ph_head.pkt_head[quabo_num] = in_block->header.pkt_head[pktIndex];

        // Mark the quabo slot as taken
        //
        ph_data->quabos_bitmap |= quabo_bit;
        ph_data->ph_head.mod_num = in_block->header.pkt_head[pktIndex].mod_num;
        ph_data->ph_head.group_ph_frames = group_ph_frames; 
    }
}


// quabo info for determining packet loss
//
typedef struct quabo_info{
    uint16_t prev_pkt_num[NUM_OF_MODES+1];
    int lost_pkts[NUM_OF_MODES+1];
} quabo_info_t;

quabo_info_t* quabo_info_t_new(){
    quabo_info_t* value = (quabo_info_t*) malloc(sizeof(struct quabo_info));
    memset(value->lost_pkts, -1, sizeof(value->lost_pkts));
    memset(value->prev_pkt_num, 0, sizeof(value->prev_pkt_num));
    return value;
}

// array of pointers to module objects
//
static MODULE_IMAGE_BUFFER* moduleInd[MAX_MODULE_INDEX] = {NULL};

//array of pointers to module PH images
//
static PH_IMAGE_BUFFER* PHmoduleInd[MAX_MODULE_INDEX] = {NULL};


// Initialization function
// is called once when the thread is created
//
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

    // Fetch user input for whether to PH frames are to be grouped.
    hgeti4(st.buf, "GROUPFRAMES", &group_ph_frames);
    if (group_ph_frames) {
        printf("Group frames is %i (True). Hashpipe will group incoming PH frames.\n", group_ph_frames);
    } else {
        printf("Group frames is %i (False). Hashpipe will not group incoming PH frames.\n", group_ph_frames);
    }

    char fbuf[100];
    char cbuf;
    unsigned int modName;

    if (modConfig_file == NULL) {
        perror("Error Opening Config File");
        exit(1);
    }
    cbuf = getc(modConfig_file);

    // Parse the Module Config file for the modules to expect data from
    // Creates structures for holding that data in the module array
    //
    while(cbuf != EOF){
        ungetc(cbuf, modConfig_file);
        if (cbuf != '#'){
            if (fscanf(modConfig_file, "%u\n", &modName) == 1){
                if (moduleInd[modName] == NULL){
                    moduleInd[modName] = new MODULE_IMAGE_BUFFER();
                    fprintf(stdout, "Created Module (Image mode): %u.%u-%u\n", 
                        (unsigned int) (modName << 2)/0x100,
                        (modName << 2) % 0x100, ((modName << 2) % 0x100) + 3
                    );
                }
                if (PHmoduleInd[modName] == NULL){
                    PHmoduleInd[modName] = new PH_IMAGE_BUFFER();
                    fprintf(stdout, "Created Module (Pulse-height): %u.%u-%u\n", 
                        (unsigned int) (modName << 2)/0x100,
                        (modName << 2) % 0x100, ((modName << 2) % 0x100) + 3
                    );
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

// function that is run once.
// To keep thread running make sure to use a while loop.
// args: Arguements passed in by the hashpipe framework

static void *run(hashpipe_thread_args_t * args){
    printf("\n---------------Running Compute Thread-----------------\n\n");
    // Local aliases to shorten access to args fields
    HSD_input_databuf_t *db_in = (HSD_input_databuf_t *)args->ibuf;
    HSD_output_databuf_t *db_out = (HSD_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    // Index values for input and output buffers
    int rv;
    uint64_t mcnt=0;
    int curblock_in=0;
    int curblock_out=0;
    int INTSIG;

    // Variables to display pkt info
    uint8_t acq_mode;
        // The current mode of the packet block
    quabo_info_t* quaboInd[0xffff] = {NULL};
        // hash table mapping quabo number to linked list ind
    quabo_info_t* currentQuabo;
        // Pointer to the quabo info that is currently being used
    uint16_t boardLoc;
        // The boardLoc(quabo index) for the current packet
    
    // Counters for the packets lost
    int total_lost_pkts = 0;
    int current_pkt_lost;
    
    while(run_threads()){
        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "COMBLKIN", curblock_in);
        hputs(st.buf, status_key, "waiting");
        hputi4(st.buf, "COMBKOUT", curblock_out);
	    hputi8(st.buf,"COMMCNT",mcnt);
        hashpipe_status_unlock_safe(&st);

        // Wait for new input block to be filled
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

        // Note processing status
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "processing packet");
        hashpipe_status_unlock_safe(&st);

        // Resetting the values in the new output block
        db_out->block[curblock_out].header.n_img_module = 0;
        db_out->block[curblock_out].header.n_ph_img = 0;
        db_out->block[curblock_out].header.INTSIG = db_in->block[curblock_in].header.INTSIG;
        INTSIG = db_in->block[curblock_in].header.INTSIG;

        uint16_t moduleNum;
        for (int i = 0; i < db_in->block[curblock_in].header.n_pkts_in_block; i++){
            //----------------CALCULATION BLOCK-----------------
            moduleNum = db_in->block[curblock_in].header.pkt_head[i].mod_num;

            if (moduleInd[moduleNum] == NULL){
                fprintf(stderr, "Detected New Module not in Config File: %u.%u\n", (unsigned int) (moduleNum << 2)/0x100, (moduleNum << 2) % 0x100);
                fprintf(stderr, "Packet skipping\n");
                continue;
            }

            storeData(
                moduleInd[moduleNum],
                PHmoduleInd[moduleNum],
                &(db_in->block[curblock_in]),
                &(db_out->block[curblock_out]),
                i
            );
            
            //------------End CALCULATION BLOCK----------------


            // Find the packet number and compute the loss of packets
            // by using packet number
            // Read the packet number from the packet
            acq_mode = db_in->block[curblock_in].header.pkt_head[i].acq_mode;
            boardLoc = db_in->block[curblock_in].header.pkt_head[i].mod_num * 4 + db_in->block[curblock_in].header.pkt_head[i].quabo_num;

            // Check if there is a quabo info for the current quabo packet.
            // If not create an object
            if (quaboInd[boardLoc] == NULL){
                quaboInd[boardLoc] = quabo_info_t_new();
                //Create a new quabo info object

                printf("New Quabo Detected ID:%u.%u\n", (boardLoc >> 8) & 0x00ff, boardLoc & 0x00ff);
            }

            // Set the current Quabo to the one stored in memory
            currentQuabo = quaboInd[boardLoc];

            // if it is newly created quabo info,
            // initialize the lost packet number to 0
            if (currentQuabo->lost_pkts[acq_mode] < 0) {
                currentQuabo->lost_pkts[acq_mode] = 0;
            } else {
                // check if the current packet number is less than the previous.
                // If so the number has overflowed and looped.
                // if this has happened, take the difference of the
                // packet numbers minus 1 to be the packets lost
                if (db_in->block[curblock_in].header.pkt_head[i].pkt_num < currentQuabo->prev_pkt_num[acq_mode]) {
                    current_pkt_lost = (0xffff - currentQuabo->prev_pkt_num[acq_mode]) + db_in->block[curblock_in].header.pkt_head[i].pkt_num;
                } else {
                    current_pkt_lost = (db_in->block[curblock_in].header.pkt_head[i].pkt_num - currentQuabo->prev_pkt_num[acq_mode]) - 1;
                }
                
                currentQuabo->lost_pkts[acq_mode] += current_pkt_lost;
                    // Add this packet lost to the total for this quabo
                total_lost_pkts += current_pkt_lost;
                    // Add this packet lost to the overall total for all quabos
            }

            // Update the previous packet number to be the current packet number
            currentQuabo->prev_pkt_num[acq_mode] = db_in->block[curblock_in].header.pkt_head[i].pkt_num;
        }

        // Update input and output block for both buffers
        // Mark output block as full and advance

        HSD_output_databuf_set_filled(db_out, curblock_out);
        curblock_out = (curblock_out + 1) % db_out->header.n_block;

        // Mark input block as free and advance
        HSD_input_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->header.n_block;
        mcnt++;

        // Break out when SIGINT is found
        if (INTSIG) {
            printf("COMPUTE_THREAD Ended\n");
            break;
        }

        // display packetnum in status

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

        // Check for cancel
        pthread_testcancel();
    }

    printf("Returned Compute_thread\n");
    return THREAD_OK;
}

// Sets the functions and buffers for this thread
//
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

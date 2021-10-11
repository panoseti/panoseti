/* HSD_compute_thread.c
 *
 * Does pre processing on the data coming from the quabos before writing to file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include "hashpipe.h"
#include "HSD_databuf.h"


#define NUM_OF_MODES 7 // Number of mode and also used the create the size of array (Modes 1,2,3,6,7)
//#define TEST_MODE


/**
 * The module ID structure that is used to store a lot of the information regarding the current pair of module.
 * Module pairs consists of PKTPERPAIR(8) quabos.
 */
typedef struct module_data {
    uint32_t upper_nanosec;
    uint32_t lower_nanosec;
    int last_mode;
    uint8_t status;
    module_header_t mod_head[QUABOPERMODULE];
    uint8_t data[MODULEDATASIZE];
    module_data(){
        this->upper_nanosec = 0;
        this->lower_nanosec = 0;
        this->last_mode = 0;
        this->status = 0;
    };
    int copy_to(module_data *mod_data) {
        mod_data->upper_nanosec = this->upper_nanosec;
        mod_data->lower_nanosec = this->lower_nanosec;
        mod_data->last_mode = this->last_mode;
        mod_data->status = this->status;
        for (int i = 0; i < QUABOPERMODULE; i++){
            this->mod_head[i].copy_to(&(mod_data->mod_head[i]));
        }
        memcpy(mod_data->data, this->data, sizeof(uint8_t)*MODULEDATASIZE);
    };
    int clear(){
        this->upper_nanosec = 0;
        this->lower_nanosec = 0;
        this->last_mode = 0;
        this->status = 0;
        for (int i = 0; i < QUABOPERMODULE; i++){
            this->mod_head[i].clear();
        }
        memset(this->data, 0, sizeof(uint8_t)*MODULEDATASIZE);
    };
    std::string toString(){
        return "";
    };
    int equal_to(module_data *mod_data){
        if (this->upper_nanosec != mod_data->upper_nanosec
            || this->lower_nanosec != mod_data->lower_nanosec
            || this->last_mode != mod_data->last_mode
            || this->status != mod_data->status){
            return 0;
        }
        for (int i = 0; i < QUABOPERMODULE; i++){
            if (!this->mod_head[i].equal_to(&(mod_data->mod_head[i]))){
                return 0;
            }
        }
        if (memcmp(this->data, mod_data->data, sizeof(uint8_t)*MODULEDATASIZE) == 0){
            return 0;
        }
        return 1;
    }
} module_data_t;

/**
 * Writes the module pair data to output buffer
 */
void write_img_to_out_buffer(module_data_t* mod_data, HSD_output_block_t* out_block){
    int out_index = out_block->header.stream_block_size;
    HSD_output_block_header_t* out_header = &(out_block->header);
    
    for (int i = 0; i < QUABOPERMODULE; i++){
        mod_data->mod_head[i].copy_to(&(out_header->img_pkt_head[out_index]));
    }
    
    memcpy(out_block->stream_block + (out_index * MODULEDATASIZE), mod_data->data, sizeof(uint8_t)*MODULEDATASIZE);

    out_block->header.stream_block_size++;
}

//TODO
/**
 * Write PH Data to output buffer's coinc block
 */
void write_coinc_to_out_buffer(HSD_input_block_t* in_block, int pktIndex, HSD_output_block_t* out_block){
    int out_index = out_block->header.coinc_block_size;

    in_block->header.pkt_head[pktIndex].copy_to(&(out_block->header.coin_pkt_head[out_index]));
    
    memcpy(out_block->coinc_block + out_index*PKTDATASIZE, in_block->data_block + pktIndex*PKTDATASIZE, sizeof(in_block->data_block[0])*PKTDATASIZE);

    out_block->header.coinc_block_size++;
}


/**
 * Storing the module data to the modulePairData from the data pointer.
 */
void storeData(module_data_t* mod_data, HSD_input_block_t* in_block, HSD_output_block_t* out_block, int pktIndex){
    int mode;
    packet_header_t *pkt_head = &(in_block->header.pkt_head[pktIndex]);
    uint32_t nanosec = pkt_head->pkt_nsec;

    uint8_t currentStatus = (0x01 << pkt_head->qua_num);

    //Check the acqmode to determine the mode in which the packet is coming in as
    if (pkt_head->acq_mode == 0x1){
        //PH Mode
        //TODO
        write_coinc_to_out_buffer(in_block, pktIndex, out_block);
        //writePHData(moduleNum, quaboNum, PKTNUM, UTC, NANOSEC, tv_sec, tv_usec, data_ptr);
        //return;
    } else if(pkt_head->acq_mode == 0x2 || pkt_head->acq_mode == 0x3){
        //16 bit Imaging mode
        mode = 16;
    } else if (pkt_head->acq_mode == 0x6 || pkt_head->acq_mode == 0x7){
        //8 bit Imaging mode
        mode = 8;
    } else {
        //Unidentified mode
        //Return and not store the packet and return an error
        printf("A new mode was identify acqmode=%X\n ", pkt_head->acq_mode);
        printf("moduleNum=%X quaboNum=%X PKTNUM=%X\n", pkt_head->mod_num, pkt_head->qua_num, pkt_head->pkt_num);
        printf("packet skipped\n");
        return;
    }

    //Setting the upper and lower bounds of NANOSEC interval that is allowed in the grouping
    if(mod_data->status == 0){
        //Empty module pair obj
        //Setting both the upper and lower NANOSEC interval to the current NANOSEC value

        mod_data->last_mode = mode;
        mod_data->upper_nanosec = nanosec;
        mod_data->lower_nanosec = nanosec;
    } else if(nanosec > mod_data->upper_nanosec){
        mod_data->upper_nanosec = nanosec;
    } else if (nanosec < mod_data->lower_nanosec){
        mod_data->lower_nanosec = nanosec;
    }

    //Check conditions to see if they are met for writing to output buffer
    //Conditions:
    //When the current location in module pair is occupied in the module pair
    //When the mode in the module pair doesen't match the new mode
    //When the NANOSEC interval superceeded the threshold that is allowed
    if ((mod_data->status & currentStatus) || mod_data->last_mode != mode || (mod_data->upper_nanosec - mod_data->lower_nanosec) > NANOSECTHRESHOLD){

        write_img_to_out_buffer(mod_data, out_block);
        
        //Resetting values in the new emptied module pair obj
        mod_data->clear();
    }

    memcpy(mod_data->data + (pkt_head->qua_num*SCIDATASIZE*(mode/8)), in_block->data_block + (pktIndex*PKTDATASIZE), sizeof(uint8_t)*SCIDATASIZE*(mode/8));
    
    in_block->header.pkt_head[pktIndex].copy_to(&(mod_data->mod_head->pkt_head[pkt_head->qua_num]));

    //Mark the status for the packet slot as taken
    mod_data->status = mod_data->status | currentStatus;
}


/**
 * Structure of the Quabo buffer stored for determining packet loss
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

static module_data_t* moduleInd[MODULEINDEXSIZE] = {NULL};

static int init(hashpipe_thread_args_t * args){
    hashpipe_status_t st = args->st;
    //Initialize the INTSIG signal within the buffer to be zero
    printf("\n\n-----------Start Setup of Compute Thread--------------\n");
    HSD_output_databuf_t *db_out = (HSD_output_databuf_t *)args->obuf;
    for (int i = 0 ; i < db_out->header.n_block; i++){
        db_out->block[i].header.INTSIG = 0;
    }

    //Initializing the Module Pairing using the config file given
    char config_location[STRBUFFSIZE];
    sprintf(config_location, CONFIGFILE_DEFAULT);
    hgets(st.buf, "CONFIG", STRBUFFSIZE, config_location);
    FILE *modConfig_file = fopen(config_location, "r");

    char fbuf[100];
    char cbuf;
    unsigned int modName;

    if (modConfig_file == NULL) {
        perror("Error Opening Config File\n");
        exit(1);
    }
    cbuf = getc(modConfig_file);

    while(cbuf != EOF){
        ungetc(cbuf, modConfig_file);
        if (cbuf != '#'){
            if (fscanf(modConfig_file, "%u\n", &modName) == 1){
                if (moduleInd[modName] == NULL){

                    moduleInd[modName] = new module_data();

                    printf("Created Module: %u.%u-%u\n", 
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
        printf("Warning: Unable to close module configuration file.\n");
    }
    printf("-----------Finished Setup of Compute Thread-----------\n\n");
    
    return 0;

    
}


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

        db_out->block[curblock_out].header.stream_block_size = 0;
        db_out->block[curblock_out].header.coinc_block_size = 0;
        db_out->block[curblock_out].header.INTSIG = db_in->block[curblock_in].header.INTSIG;
        INTSIG = db_in->block[curblock_in].header.INTSIG;

        uint16_t moduleNum;
        #ifdef TEST_MODE
            printf("Size of intput buffer data block: %i\n", db_in->block[curblock_in].header.data_block_size);
        #endif
        for(int i = 0; i < db_in->block[curblock_in].header.data_block_size; i++){
            //----------------CALCULATION BLOCK-----------------
            moduleNum = db_in->block[curblock_in].header.pkt_head[i].mod_num;

            if (moduleInd[moduleNum] == NULL){

                printf("Detected New Module not in Config File: %u.%u\n", (unsigned int) (moduleNum << 2)/0x100, (moduleNum << 2) % 0x100);
                printf("Packet skipping\n");
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
            /*hputs(st.buf, "QUABOKEY", boardLocstr);
            hputi4(st.buf, "M1PKTNUM", currentQuabo->pkt_num[1]);
            hputi4(st.buf, "M2PKTNUM", currentQuabo->pkt_num[2]);
            hputi4(st.buf, "M3PKTNUM", currentQuabo->pkt_num[3]);
            hputi4(st.buf, "M6PKTNUM", currentQuabo->pkt_num[6]);
            hputi4(st.buf, "M7PKTNUM", currentQuabo->pkt_num[7]);*/

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
    name: "HSD_compute_thread",
    skey: "COMPUTESTAT",
    init: init,
    run: run,
    ibuf_desc: {HSD_input_databuf_create},
    obuf_desc: {HSD_output_databuf_create}
};

static __attribute__((constructor)) void ctor(){
    register_hashpipe_thread(&HSD_compute_thread);
}

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
typedef struct modulePairData {
    uint8_t status;   // Determine the which part of the data is filled 0:neither filled 1:First rank filled 2: Second rank filled
    unsigned int mod1Name;
    unsigned int mod2Name;
    uint32_t upperNANOSEC;
    uint32_t lowerNANOSEC;
    int lastMode;
    uint16_t PKTNUM[PKTPERPAIR];
    uint32_t NANOSEC[PKTPERPAIR];
    long int tv_sec[PKTPERPAIR];
    long int tv_usec[PKTPERPAIR];
    uint8_t data[MODPAIRDATASIZE];
    modulePairData* next_moduleID;
} modulePairData_t;

/**
 * Creating a new module ID object given the ID values and module numbers.
 */
modulePairData_t* modulePairData_t_new(unsigned int mod1, unsigned int mod2){
    modulePairData_t* value = (modulePairData_t*) malloc(sizeof(struct modulePairData));
    if (value == NULL){
        printf("Error: Unable to malloc space for ModulePairData\n");
        exit(1);
    }
    value->status = 0;
    value->mod1Name = mod1;
    value->mod2Name = mod2;
    value->next_moduleID = NULL;
    value->upperNANOSEC = 0;
    value->lowerNANOSEC = 0;
    return value;
}

/**
 * Creating a new module ID with zeroed/null values
 */
modulePairData_t* modulePairData_t_new(){
    return modulePairData_t_new(-1,-1);
}

/**
 * Creating a new module ID with only 1 module for when new module is detected and not in module pair config file
 */
modulePairData_t* modulePairData_t_new(unsigned mod1){
    return modulePairData_t_new(mod1, -1);
}

/**
 * Store the data from the data_ptr to the moduleData based on the mode.
 */
/*void storePktDataIntoModPair(uint8_t* moduleData, char* data_ptr, int mode, int quaboIndex){
    //memcpy(moduleData + (quaboIndex*SCIDATASIZE*(mode/8)), data_ptr, sizeof(uint8_t)*SCIDATASIZE*(mode/8));
    uint8_t *data;
    if (mode == 16){
        data = moduleData + (quaboIndex*SCIDATASIZE*2);
        for(int i = 0; i < SCIDATASIZE*2; i++){
            data[i] = data_ptr[i];
        }
    } else if(mode == 8){
        data = moduleData + (quaboIndex*SCIDATASIZE);
        for(int i = 0; i < SCIDATASIZE; i++){
            data[i] = data_ptr[i];
        }
    }
}*/

/**
 * Writes the module pair data to output buffer
 */
void writeDataToOutBuf(modulePairData_t* modulePair, HSD_output_block_t* out_block){
    int out_index = out_block->header.stream_block_size;
    HSD_output_block_header_t* out_header = &(out_block->header);

    out_header->modNum[out_index*2] = modulePair->mod1Name;
    out_header->modNum[(out_index*2)+1] = modulePair->mod2Name;

    out_header->acqmode[out_index] = modulePair->lastMode;
    
    memcpy(out_block->header.pktNum + (out_index * PKTPERPAIR), modulePair->PKTNUM, sizeof(modulePair->PKTNUM[0])*PKTPERPAIR);
    memcpy(out_block->header.pktNSEC + (out_index * PKTPERPAIR), modulePair->NANOSEC, sizeof(modulePair->NANOSEC[0])*PKTPERPAIR);
    memcpy(out_block->header.tv_sec + (out_index * PKTPERPAIR), modulePair->tv_sec, sizeof(modulePair->tv_sec[0])*PKTPERPAIR);
    memcpy(out_block->header.tv_usec + (out_index * PKTPERPAIR), modulePair->tv_usec, sizeof(modulePair->tv_usec[0])*PKTPERPAIR);
    memcpy(out_block->header.status + out_index, &(modulePair->status), sizeof(modulePair->status));
    
    memcpy(out_block->stream_block + (out_index * MODPAIRDATASIZE), modulePair->data, sizeof(uint8_t)*MODPAIRDATASIZE);

    out_block->header.stream_block_size++;
}

/**
 * Write PH Data to output buffer's coinc block
 */
void writePHToOutBuf(HSD_input_block_t* in_block, int pktIndex, HSD_output_block_t* out_block){
    int out_index = out_block->header.coinc_block_size;
    out_block->header.coin_acqmode[out_index] = in_block->header.acqmode[pktIndex];
    out_block->header.coin_pktNum[out_index] = in_block->header.pktNum[pktIndex];
    out_block->header.coin_modNum[out_index] = in_block->header.modNum[pktIndex];
    out_block->header.coin_quaNum[out_index] = in_block->header.quaNum[pktIndex];
    out_block->header.coin_pktUTC[out_index] = in_block->header.pktUTC[pktIndex];
    out_block->header.coin_pktNSEC[out_index] = in_block->header.pktNSEC[pktIndex];
    out_block->header.coin_tv_sec[out_index] = in_block->header.tv_sec[pktIndex];
    out_block->header.coin_tv_usec[out_index] = in_block->header.tv_usec[pktIndex];
    
    memcpy(out_block->coinc_block + out_index*PKTDATASIZE, in_block->data_block + pktIndex*PKTDATASIZE, sizeof(in_block->data_block[0])*PKTDATASIZE);

    out_block->header.coinc_block_size++;
}


/**
 * Storing the module data to the modulePairData from the data pointer.
 */
void storeData(modulePairData_t* module, HSD_input_block_t* in_block, HSD_output_block_t* out_block, int pktIndex){
    int mode;
    int quaboIndex;
    char acqmode = in_block->header.acqmode[pktIndex];
    uint16_t modNum = in_block->header.modNum[pktIndex];
    uint8_t quaboNum = in_block->header.quaNum[pktIndex];
    uint16_t PKTNUM = in_block->header.pktNum[pktIndex];
    uint32_t NANOSEC = in_block->header.pktNSEC[pktIndex];

    uint8_t currentStatus = (0x01 << quaboNum);

    //Check the acqmode to determine the mode in which the packet is coming in as
    if (acqmode == 0x1){
        //PH Mode
        //TODO
        writePHToOutBuf(in_block, pktIndex, out_block);
        //writePHData(moduleNum, quaboNum, PKTNUM, UTC, NANOSEC, tv_sec, tv_usec, data_ptr);
        //return;
    } else if(acqmode == 0x2 || acqmode == 0x3){
        //16 bit Imaging mode
        mode = 16;
    } else if (acqmode == 0x6 || acqmode == 0x7){
        //8 bit Imaging mode
        mode = 8;
    } else {
        //Unidentified mode
        //Return and not store the packet and return an error
        printf("A new mode was identify acqmode=%X\n", acqmode);
        printf("moduleNum=%X quaboNum=%X PKTNUM=%X\n", modNum, quaboNum, PKTNUM);
        printf("packet skipped\n");
        return;
    }

    //Set the Index where the packet would be stored within the module pair
    quaboIndex = quaboNum;

    if(modNum == module->mod2Name){
        currentStatus = currentStatus << 4;
        quaboIndex += 4;
    }

    //Setting the upper and lower bounds of NANOSEC interval that is allowed in the grouping
    if(module->status == 0){
        //Empty module pair obj
        //Setting both the upper and lower NANOSEC interval to the current NANOSEC value
        module->lastMode = mode;
        module->upperNANOSEC = NANOSEC;
        module->lowerNANOSEC = NANOSEC;
    } else if(NANOSEC > module->upperNANOSEC){
        module->upperNANOSEC = NANOSEC;
    } else if (NANOSEC < module->lowerNANOSEC){
        module->lowerNANOSEC = NANOSEC;
    }

    //Check conditions to see if they are met for writing to file
    //Conditions:
    //When the current location in module pair is occupied in the module pair
    //When the mode in the module pair doesen't match the new mode
    //When the NANOSEC interval superceeded the threshold that is allowed
    if ((module->status & currentStatus) || module->lastMode != mode || (module->upperNANOSEC - module->lowerNANOSEC) > NANOSECTHRESHOLD){

        writeDataToOutBuf(module, out_block);

        memset(module->PKTNUM, 0, sizeof(uint16_t)*PKTPERPAIR);
        memset(module->NANOSEC, 0, sizeof(uint32_t)*PKTPERPAIR);
        memset(module->tv_sec, 0 , sizeof(long)*PKTPERPAIR);
        memset(module->tv_usec, 0, sizeof(long)*PKTPERPAIR); 
        
        //Resetting values in the new emptied module pair obj
        module->status = 0;
        module->lastMode = mode;
        module->upperNANOSEC = NANOSEC;
        module->lowerNANOSEC = NANOSEC;
    }

    //printf("ACQMode = %u, LastMode = %u, Mode = %u, ModuleNum = %u, QuaboNum = %u, UTC = %u, NANOSEC = %u, PKTNUM = %u\n", acqmode, module->lastMode, mode, moduleNum, quaboNum, UTC, NANOSEC, PKTNUM);
    //storePktDataIntoModPair((uint8_t *)module->data, data_ptr, mode, quaboIndex);
    memcpy(module->data + (quaboIndex*SCIDATASIZE*(mode/8)), in_block->data_block + (pktIndex*PKTDATASIZE), sizeof(uint8_t)*SCIDATASIZE*(mode/8));
    module->lastMode = mode;
    module->PKTNUM[quaboIndex] = PKTNUM;
    //module->UTC[quaboIndex] = UTC;
    module->tv_sec[quaboIndex] = in_block->header.tv_sec[pktIndex];
    module->tv_usec[quaboIndex] = in_block->header.tv_usec[pktIndex];
    module->NANOSEC[quaboIndex] = NANOSEC;

    //Mark the status for the packet slot as taken
    module->status = module->status | currentStatus;
}


/**
 * Structure of the Quabo buffer stored for determining packet loss
 */
typedef struct quabo_info{
    uint16_t prev_pkt_num[NUM_OF_MODES+1];
    int lost_pkts[NUM_OF_MODES+1];
    quabo_info* next_quabo_info;
} quabo_info_t;

/**
 * Initializing a new quabo_info object
 */
quabo_info_t* quabo_info_t_new(){
    quabo_info_t* value = (quabo_info_t*) malloc(sizeof(struct quabo_info));
    memset(value->lost_pkts, -1, sizeof(value->lost_pkts));
    memset(value->prev_pkt_num, 0, sizeof(value->prev_pkt_num));
    value->next_quabo_info = NULL;
    return value;
}

//Initializing the linked list to be used for stroing the modulePairData
static modulePairData_t* moduleListBegin = modulePairData_t_new();
static modulePairData_t* moduleListEnd = moduleListBegin;
static modulePairData_t* moduleInd[MODULEINDEXSIZE] = {NULL};



static int init(hashpipe_thread_args_t * args){
    //Initialize the INTSIG signal within the buffer to be zero
    printf("\n\n-----------Start Setup of Compute Thread--------------\n");
    HSD_output_databuf_t *db_out = (HSD_output_databuf_t *)args->obuf;
    for (int i = 0 ; i < db_out->header.n_block; i++){
        db_out->block[i].header.INTSIG = 0;
    }

    //Initializing the Module Pairing using the config file given
    FILE *modConfig_file = fopen(CONFIGFILE, "r");
    char fbuf[100];
    char cbuf;
    unsigned int mod1Name;
    unsigned int mod2Name;

    if (modConfig_file == NULL) {
        perror("Error Opening Config File\n");
        exit(1);
    }
    cbuf = getc(modConfig_file);
    char moduleName[50];

    while(cbuf != EOF){
        ungetc(cbuf, modConfig_file);
        if (cbuf != '#'){
            if (fscanf(modConfig_file, "%u %u\n", &mod1Name, &mod2Name) == 2){
                if (moduleInd[mod1Name] == NULL && moduleInd[mod2Name] == NULL){

                    sprintf(moduleName, MODULEPAIR_FORMAT, mod1Name, mod2Name);

                    moduleInd[mod1Name] = moduleInd[mod2Name] = moduleListEnd->next_moduleID 
                                            = modulePairData_t_new(mod1Name, mod2Name);
                    
                    moduleListEnd = moduleListEnd->next_moduleID;
                    
                    //createQuaboTables(moduleListEnd->dynamicMeta, moduleListEnd);

                    printf("Created Module Pair: %u.%u-%u and %u.%u-%u\n", 
                    (unsigned int) (mod1Name << 2)/0x100, (mod1Name << 2) % 0x100, ((mod1Name << 2) % 0x100) + 3,
                    (mod2Name << 2)/0x100, (mod2Name << 2) % 0x100, ((mod2Name << 2) % 0x100) + 3);
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
    quabo_info_t* quaboListBegin = quabo_info_t_new();  //Initializing the quabo info linked list
    quabo_info_t* quaboListEnd = quaboListBegin;        //Setting the pointer to be the end of the linked list
    quabo_info_t* quaboInd[0xffff] = {NULL};            //Create a rudimentary hash map of the quabo number and linked list ind

    quabo_info_t* currentQuabo;                         //Pointer to the quabo info that is currently being used
    uint16_t boardLoc;                                  //The boardLoc(quabo index) for the current packet
    char* boardLocstr = (char *)malloc(sizeof(char)*10);

    #ifdef TEST_MODE
        FILE *fptr;
        HSD_output_block_header_t* outBlockHeader;
        fptr = fopen("./output_buffer.log", "w");
        fprintf(fptr, "%s%15s%15s%15s%15s%15s%15s%15s\n",
                "ACQMODE", "PKTNUM", "MODNUM", "QUABONUM", "PKTUTC", "PKTNSEC", "tv_sec", "tv_usec");
    #endif
    
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

        modulePairData_t* currentModule;
        uint16_t moduleNum;
        #ifdef TEST_MODE
            printf("Size of intput buffer data block: %i\n", db_in->block[curblock_in].header.data_block_size);
        #endif
        for(int i = 0; i < db_in->block[curblock_in].header.data_block_size; i++){
            //----------------CALCULATION BLOCK-----------------
            moduleNum = db_in->block[curblock_in].header.modNum[i];

            if (moduleInd[moduleNum] == NULL){

                printf("Detected New Module not in Config File: %u.%u\n", (unsigned int) (moduleNum << 2)/0x100, (moduleNum << 2) % 0x100);
                printf("Packet skipping\n");
                continue;

            } else {
                currentModule = moduleInd[moduleNum];
            }

            storeData(currentModule, &(db_in->block[curblock_in]), &(db_out->block[curblock_out]), i);
            
            //------------End CALCULATION BLOCK----------------


            //Finding the packet number and computing the lost of packets by using packet number
            //Read the packet number from the packet
            mode = db_in->block[curblock_in].header.acqmode[i];
            boardLoc = db_in->block[curblock_in].header.modNum[i] * 4 + db_in->block[curblock_in].header.quaNum[i];

            //Check to see if there is a quabo info for the current quabo packet. If not create an object
            if (quaboInd[boardLoc] == NULL){
                quaboInd[boardLoc] = quabo_info_t_new();            //Create a new quabo info object

                printf("New Quabo Detected ID:%u.%u\n", (boardLoc >> 8) & 0x00ff, boardLoc & 0x00ff); //Output the terminal the new quabo

                quaboListEnd->next_quabo_info = quaboInd[boardLoc];
                quaboListEnd = quaboListEnd->next_quabo_info;       //Append the new quabo info to the end of the linked list
            }

            //Set the current Quabo to the one stored in memory
            currentQuabo = quaboInd[boardLoc];

            //Check to see if it is newly created quabo info if so then inialize the lost packet number to 0
            if (currentQuabo->lost_pkts[mode] < 0) {
                currentQuabo->lost_pkts[mode] = 0;
            } else {
                //Check to see if the current packet number is less than the previous. If so the number has overflowed and looped.
                //Compenstate for this if this has happend, and then take the difference of the packet numbers minus 1 to be the packets lost
                if (db_in->block[curblock_in].header.pktNum[i] < currentQuabo->prev_pkt_num[mode])
                    current_pkt_lost = (0xffff - currentQuabo->prev_pkt_num[mode]) + db_in->block[curblock_in].header.pktNum[i];
                else
                    current_pkt_lost = (db_in->block[curblock_in].header.pktNum[i] - currentQuabo->prev_pkt_num[mode]) - 1;
                
                currentQuabo->lost_pkts[mode] += current_pkt_lost; //Add this packet lost to the total for this quabo
                total_lost_pkts += current_pkt_lost;               //Add this packet lost to the overall total for all quabos
            }
            currentQuabo->prev_pkt_num[mode] = db_in->block[curblock_in].header.pktNum[i]; //Update the previous packet number to be the current packet number

            /*
            //Copy to output buffer
            if (mode < 4){
                memcpy(db_out->block[curblock_out].stream_block+i*PKTDATASIZE, db_in->block[curblock_in].data_block+i*PKTDATASIZE, PKTDATASIZE*sizeof(unsigned char));
            } else {
                memcpy(db_out->block[curblock_out].stream_block+i*PKTDATASIZE, db_in->block[curblock_in].data_block+i*PKTDATASIZE, BIT8PKTDATASIZE*sizeof(unsigned char));
            }

            //Copy time over to output
            db_out->block[curblock_out].header.acqmode[i] = db_in->block[curblock_in].header.acqmode[i];
            db_out->block[curblock_out].header.pktNum[i] = db_in->block[curblock_in].header.pktNum[i];
            db_out->block[curblock_out].header.modNum[i] = db_in->block[curblock_in].header.modNum[i];
            db_out->block[curblock_out].header.quaNum[i] = db_in->block[curblock_in].header.quaNum[i];
            db_out->block[curblock_out].header.pktUTC[i] = db_in->block[curblock_in].header.pktUTC[i];
            db_out->block[curblock_out].header.pktNSEC[i] = db_in->block[curblock_in].header.pktNSEC[i];


            db_out->block[curblock_out].header.tv_sec[i] = db_in->block[curblock_in].header.tv_sec[i];
            db_out->block[curblock_out].header.tv_usec[i] = db_in->block[curblock_in].header.tv_usec[i];

            db_out->block[curblock_out].header.stream_block_size++;*/        
        }

        

        #ifdef TEST_MODE
            outBlockHeader = &(db_out->block[curblock_out].header);
            for (int i = 0; i < outBlockHeader->stream_block_size; i++){
                fprintf(fptr, "%7u%15u%15u%15u%15u%15u%15lu%15lu\n",
                        outBlockHeader->acqmode[i], outBlockHeader->pktNum[i],
                        outBlockHeader->modNum[i], outBlockHeader->quaNum[i],
                        outBlockHeader->pktUTC[i], outBlockHeader->pktNSEC[i],
                        outBlockHeader->tv_sec[i], outBlockHeader->tv_usec[i]);
            }
        #endif

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

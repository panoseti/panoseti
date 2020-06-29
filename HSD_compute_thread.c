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

uint16_t findPktNum(char data1, char data2){
    return ((data2 << 8) & 0xff00) | ((data1) & 0x00ff);
}

static void *run(hashpipe_thread_args_t * args){
    // Local aliases to shorten access to args fields
    HSD_input_databuf_t *db_in = (HSD_input_databuf_t *)args->ibuf;
    HSD_output_databuf_t *db_out = (HSD_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    int rv;
    uint64_t mcnt=0;
    int curblock_in=0;
    int curblock_out=0;

    //Variables to display pkt info
    uint8_t mode;
    uint16_t pkt_num[NUM_OF_MODES+1] = {0};
    uint16_t prev_pkt_num[NUM_OF_MODES+1] = {0};
    int lost_pkts[NUM_OF_MODES+1];
    memset(lost_pkts, -1, sizeof(lost_pkts));
    int total_lost_pkts = 0;
    int current_pkt_lost;
    //Compute Elements
    char *str_q;
    str_q = (char *)malloc(BLOCKSIZE*sizeof(unsigned char));

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

        //CALCULATION BLOCK
        //TODO
        //Get data from buffer
        memcpy(str_q, db_in->block[curblock_in].data_block, BLOCKSIZE*sizeof(unsigned char));

        for(int i = 0; i < N_PKT_PER_BLOCK; i++){
            //Read the packet number from the packet
            mode = str_q[i*PKTSIZE];
            pkt_num[mode] = findPktNum(str_q[i*PKTSIZE+2], str_q[i*PKTSIZE+3]);

            #ifdef TEST_MODE
                printf("pkt_num:%i\n", pkt_num[mode]);
                printf("lost_pkt:%i\n\n", lost_pkts[mode]);
            #endif
            //Check to see if the next packet is 1 more than the previous packet
            if (lost_pkts[mode] < 0) {
                lost_pkts[mode] = 0;
            } else {
                if (pkt_num[mode] < prev_pkt_num[mode])
                    current_pkt_lost = (0xffff - prev_pkt_num[mode]) + pkt_num[mode] - 1;
                else 
                    current_pkt_lost = (pkt_num[mode] - prev_pkt_num[mode]) - 1;
                
                lost_pkts[mode] += current_pkt_lost;
                total_lost_pkts += current_pkt_lost;
            }
            prev_pkt_num[mode] = pkt_num[mode];
            
        }

        //Copy the input packet to the output packet
        memcpy(db_out->block[curblock_out].result_block, str_q, BLOCKSIZE*sizeof(unsigned char));

        /*Update input and output block for both buffers*/
        //Mark output block as full and advance
        HSD_output_databuf_set_filled(db_out, curblock_out);
        curblock_out = (curblock_out + 1) % db_out->header.n_block;

        //Mark input block as free and advance
        HSD_input_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->header.n_block;
        mcnt++;

        //display packetnum in status
        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "M1PKTNUM", pkt_num[1]);
        hputi4(st.buf, "M2PKTNUM", pkt_num[2]);
        hputi4(st.buf, "M3PKTNUM", pkt_num[3]);
        hputi4(st.buf, "M6PKTNUM", pkt_num[6]);
        hputi4(st.buf, "M7PKTNUM", pkt_num[7]);

        hputi4(st.buf, "TPKTLST", total_lost_pkts);
        hputi4(st.buf, "M1PKTLST", lost_pkts[1]);
        hputi4(st.buf, "M2PKTLST", lost_pkts[2]);
        hputi4(st.buf, "M3PKTLST", lost_pkts[3]);
        hputi4(st.buf, "M6PKTLST", lost_pkts[6]);
        hputi4(st.buf, "M7PKTLST", lost_pkts[7]);
        hashpipe_status_unlock_safe(&st);

        //Check for cancel
        pthread_testcancel();

    }

    //printf("\n");
    return THREAD_OK;
}

static hashpipe_thread_desc_t HSD_compute_thread = {
    name: "HSD_compute_thread",
    skey: "COMPUTESTAT",
    init: NULL,
    run: run,
    ibuf_desc: {HSD_input_databuf_create},
    obuf_desc: {HSD_output_databuf_create}
};

static __attribute__((constructor)) void ctor(){
    register_hashpipe_thread(&HSD_compute_thread);
}

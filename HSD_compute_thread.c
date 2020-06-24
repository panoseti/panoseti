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
    uint16_t pkt_num;
    uint16_t prev_pkt_num;
    int lost_pkts = -1;
    //Compute Elements
    char *str_q;
    str_q = (char *)malloc(BLOCKSIZE*sizeof(char));

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
        memcpy(str_q, db_in->block[curblock_in].data_block, BLOCKSIZE*sizeof(char));

        //Read the packet number from the packet
        //pkt_num = findPktNum(str_q[2], str_q[3]);

        //Check to see if the next packet is 1 more than the previous packet
        /*if (lost_pkts < 0)
            lost_pkts = 0;
        else
            if (pkt_num < prev_pkt_num)
                lost_pkts += (0xffff - prev_pkt_num) + pkt_num - 1;
            else 
                lost_pkts += (pkt_num - prev_pkt_num) - 1;
        prev_pkt_num = pkt_num;

        printf("Lost Packets %i\n", lost_pkts);*/

        //Copy the input packet to the output packet
        memcpy(db_out->block[curblock_out].result_block, str_q, BLOCKSIZE*sizeof(char));

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
        hputi4(st.buf, "PKTNUM", pkt_num);
        hputi4(st.buf, "PKTLOST", lost_pkts);
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

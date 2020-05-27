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

static uint8_t char_to_uint8_flipped(char ch){
    switch (ch){
        case '0': return 0x00;
        case '1': return 0x08;
        case '2': return 0x04;
        case '3': return 0x0c;
        case '4': return 0x02;
        case '5': return 0x0a;
        case '6': return 0x06;
        case '7': return 0x0e;
        case '8': return 0x01;
        case '9': return 0x09;
        case 'a': return 0x05;
        case 'b': return 0x0d;
        case 'c': return 0x03;
        case 'd': return 0x0b;
        case 'e': return 0x07;
        case 'f': return 0x0f;
        default: return 0x00;
    }
}
uint8_t hex_to_uint8(char first, char second){
    return (char_to_uint8_flipped(first)<<4) + char_to_uint8_flipped(second);
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

    //TODO: Temporarily display packet number
    uint8_t pkt_num;
    //Compute Elements
    char *str_q;
    str_q = (char *)malloc(PKTSIZE*sizeof(char));

    while(run_threads()){
        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "GPUBLKIN", curblock_in);
        hputs(st.buf, status_key, "waiting");
        hputi4(st.buf, "GPUBKOUT", curblock_out);
	    hputi8(st.buf,"GPUMCNT",mcnt);
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
        memcpy(str_q, db_in->block[curblock_in].packet_bytes, PKTSIZE);

        //Read the packet number from the packet
        pkt_num = hex_to_uint8(str_q[2], str_q[3]);
        printf("Packet number %i is being processed", pkt_num);

        //Copy the input packet to the output packet
        memcpy(db_out->block[curblock_out].packet_result, str_q, PKTSIZE);

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
        hashpipe_status_unlock_safe(&st);

        //Check for cancel
        pthread_testcancel();

    }
    return THREAD_OK;
}

static hashpipe_thread_desc_t HSD_compute_thread = {
    name: "HSD_compute_thread",
    skey: "GPUSTAT",
    init: NULL,
    run: run,
    ibuf_desc: {HSD_input_databuf_create},
    obuf_desc: {HSD_output_databuf_create}
};

static __attribute__((constructor)) void ctor(){
    register_hashpipe_thread(&HSD_compute_thread);
}

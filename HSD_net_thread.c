/*
 * HSD_net_thread.c
 * 
 * The net thread which is used to read packets from the quabos.
 * These packets are then written into the shared memory blacks,
 * which then allows for the pre-process of the data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include "hashpipe.h"
#include "HSD_databuf.h"

//defining a struct of type hashpipe_udp_params as defined in hashpipe_udp.h
static struct hashpipe_udp_params params;

static int init(hashpipe_thread_args_t * args){
    hashpipe_status_t st = args->st;
    strcpy(params.bindhost, "127.0.0.1");
    //selecting a port to listen to
    params.bindport = 60001;
    params.packet_size = 0;
    hashpipe_udp_init(&params);
    hashpipe_status_lock_safe(&st);
    hputi8(st.buf, "NPACKETS", 0);
    hputi8(st.buf, "NBYTES", 0);
    hashpipe_status_unlock_safe(&st);
    return 0;
}

static void *run(hashpipe_thread_args_t * args){
    HSD_input_databuf_t *db  = (HSD_input_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    /* Main loop */
    int rv, n;
    uint64_t mcnt = 0;
    int block_idx = 0;

    //Input elements(Packets from Quabo)
    char *str_rcv, *str_q;
    str_rcv = (char *)malloc(PKTSIZE*sizeof(char));
    str_q = (char *)malloc(PKTSIZE*sizeof(char));

    uint64_t npackets = 0;
    uint64_t nbytes = 0;
    int hasData;


    while(run_threads()){
        //Update the info of the buffer
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "waiting");
        hputi4(st.buf, "NETBKOUT", block_idx);
        hputi8(st.buf,"NETMCNT",mcnt);
        hputi8(st.buf, "NPACKETS", npackets);
        hputi8(st.buf, "NBYTES", nbytes);
        hashpipe_status_unlock_safe(&st);

        // Wait for data
        /* Wait for new block to be free, then clear it
            * if necessary and fill its header with new values.
            */
        while ((rv=HSD_input_databuf_wait_free(db, block_idx)) != HASHPIPE_OK) {
          if (rv==HASHPIPE_TIMEOUT) {
                //Setting the statues of the buffer as blocked.
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked");
                hashpipe_status_unlock_safe(&st);
                continue;
          } else {
                hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                pthread_exit(NULL);
                break;
          }
        }

        //Updating the progress of the buffer to be recieving
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "receiving");
        hashpipe_status_unlock_safe(&st);
        n = recv(params.sock, str_rcv, PKTSIZE*sizeof(char), 0);//recvfrom(params.sock, str_rcv, PKTSIZE*sizeof(char), 0,0,0);
        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "N_VALUE", n);
        hputs(st.buf, "PKTError", strerror(errno));
        hashpipe_status_unlock_safe(&st);

        //if recieved packet has data;
        if (n > 0){
            npackets++;
            nbytes += n;
            str_q = str_rcv;
            hasData = 1;
        } else {
            hasData = 0;
        }

        //if there is data that has been recieved
        if (hasData){

            //move these headers and packet to buffer
            db->block[block_idx].header.mcnt = mcnt;
            memcpy(db->block[block_idx].packet_bytes, str_q, PKTSIZE*sizeof(char));

            //Mark block as full
            if(HSD_input_databuf_set_filled(db, block_idx) != HASHPIPE_OK){
                hashpipe_error(__FUNCTION__, "error waiting for databuf filled call");
                pthread_exit(NULL);
            }
        } else {
            continue;
        }

        //Setup for next block
        block_idx = (block_idx + 1) % db->header.n_block;

        //Will exit if thread has been cancelled
        pthread_testcancel();
    }

    //Thread success!
    return THREAD_OK;
}

static hashpipe_thread_desc_t HSD_net_thread = {
    name: "HSD_net_thread",
    skey: "NETSTAT",
    init: init,
    run: run,
    ibuf_desc: {NULL},
    obuf_desc: {HSD_input_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&HSD_net_thread);
}

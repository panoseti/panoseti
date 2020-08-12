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

#define PKTSOCK_BYTES_PER_FRAME (16384)
#define PKTSOCK_FRAMES_PER_BLOCK (8)
#define PKTSOCK_NBLOCKS (20)
#define PKTSOCK_NFRAMES (PKTSOCK_FRAMES_PER_BLOCK * PKTSOCK_NBLOCKS)
#define TEST_MODE

//defining a struct of type hashpipe_udp_params as defined in hashpipe_udp.h
//static struct hashpipe_udp_params params;

static int init(hashpipe_thread_args_t * args){

    // define network params
    char bindhost[80];
    int bindport = 60001;
    hashpipe_status_t st = args->st;
    strcpy(bindhost, "0.0.0.0");

    //selecting a port to listen to
    //params.bindport = 60001;
    //params.packet_size = 0;
    //hashpipe_udp_init(&params);
    hashpipe_status_lock_safe(&st);
    // Get info from status buffer if present
    hgets(st.buf, "BINDHOST", 80, bindhost);
    hgeti4(st.buf, "BINDPORT", &bindport);

    //Store bind host/port info and other info in status buffer
    hputs(st.buf, "BINDHOST", bindhost);
	hputi4(st.buf, "BINDPORT", bindport);
    hputi8(st.buf, "NPACKETS", 0);
    hashpipe_status_unlock_safe(&st);

    // Set up pktsocket
    struct hashpipe_pktsock *p_ps = (struct hashpipe_pktsock *)
    malloc(sizeof(struct hashpipe_pktsock));

    if(!p_ps) {
        perror(__FUNCTION__);
        return -1;
    }

    /* Make frame_size be a divisor of block size so that frames will be
	contiguous in mapped mempory.  block_size must also be a multiple of
	page_size.  Easiest way is to oversize the frames to be 16384 bytes, which
	is bigger than we need, but keeps things easy. */
	p_ps->frame_size = PKTSOCK_BYTES_PER_FRAME;
	// total number of frames
	p_ps->nframes = PKTSOCK_NFRAMES;
	// number of blocks
	p_ps->nblocks = PKTSOCK_NBLOCKS;

    int rv = hashpipe_pktsock_open(p_ps, bindhost, PACKET_RX_RING);
	if (rv!=HASHPIPE_OK) {
        hashpipe_error("HSD_net_thread", "Error opening pktsock.");
        pthread_exit(NULL);
	}

    // Store packet socket pointer in args
	args->user_data = p_ps;
	// Success!
    return 0;
}

typedef struct {
    int mode;
    uint64_t packetNum;
    int boardLocation;
}packet_header_t;

static inline void get_header(unsigned char *p_frame, packet_header_t *pkt_header) {
    uint64_t raw_header;
    raw_header = *(unsigned long long *)(PKT_UDP_DATA(p_frame));
    pkt_header->mode = (raw_header & 0x00000000000000ff);
    pkt_header->packetNum = (raw_header & 0x00000000ffff0000) >> (16);
    pkt_header->boardLocation = (raw_header & 0x0000ffff00000000) >> (32);

    #ifdef TEST_MODE
        //printf("Mode:%i ", pkt_header->mode);
        //printf("BoardLocation:%02X ", pkt_header->boardLocation);
        //printf("Packet:%lu\n", pkt_header->packetNum);
        //printf("Header:%02lx\n", (raw_header & 0xffffffffffffffff));
    #endif
}

static void *run(hashpipe_thread_args_t * args){
    HSD_input_databuf_t *db  = (HSD_input_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    /* Main loop */
    int rv, n;
    uint64_t mcnt = 0;
    int block_idx = 0;
    packet_header_t pkt_header;
            
    //Input elements(Packets from Quabo)
    char *str_rcv, *str_q;
    str_rcv = (char *)malloc(PKTSIZE*sizeof(char));
    str_q = (char *)malloc(PKTSIZE*sizeof(char));

    //Compute the pkt_loss in the compute thread
    /*uint64_t pkt_loss = 0;          // number of packets has been lost
    uint64_t pkt_loss_rate = 0;     // packets lost rate*/
    unsigned int pktsock_pkts = 0;      // Stats counter from socket packet
    unsigned int pktsock_drops = 0;     // Stats counter from socket packet
    uint64_t npackets = 0;          //number of received packets
    int bindport = 0;

    hashpipe_status_lock_safe(&st);
	// Get info from status buffer if present (no change if not present)
	hgeti4(st.buf, "BINDPORT", &bindport);
	hputs(st.buf, status_key, "running");
	hashpipe_status_unlock_safe(&st);

    // Get pktsock from args
	struct hashpipe_pktsock * p_ps = (struct hashpipe_pktsock*)args->user_data;
	pthread_cleanup_push(free, p_ps);
	pthread_cleanup_push((void (*)(void *))hashpipe_pktsock_close, p_ps);

	// Drop all packets to date
	unsigned char *p_frame;
	while(p_frame=hashpipe_pktsock_recv_frame_nonblock(p_ps)) {
		hashpipe_pktsock_release_frame(p_frame);
	}


    while(run_threads()){
        //Update the info of the buffer
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "waiting");
        hputi4(st.buf, "NETBKOUT", block_idx);
        hputi8(st.buf,"NETMCNT",mcnt);
        hputi8(st.buf, "NPACKETS", npackets);
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

        for (int i = 0; i < N_PKT_PER_BLOCK; i++){
            //printf("For loop i: %i\n", i);
            do {
                p_frame = hashpipe_pktsock_recv_udp_frame_nonblock(p_ps, bindport);
            } 
            while (!p_frame && run_threads());

            if(!run_threads()) break;
            //TODO
            //Check Packet Number at the beginning and end to see if we lost any packets
            npackets++;
            get_header(p_frame, &pkt_header);
            
            //printf("PKT_UDP_DATA %i\n", PKT_UDP_DATA(p_frame)-8);

            hashpipe_pktsock_release_frame(p_frame);

            //move these headers and packet to buffer
            if (pkt_header.mode < 4){
                memcpy(db->block[block_idx].data_block+i*PKTSIZE, PKT_UDP_DATA(p_frame), PKTSIZE*sizeof(unsigned char));
            } else {
                memcpy(db->block[block_idx].data_block+i*PKTSIZE, PKT_UDP_DATA(p_frame), BIT8PKTSIZE*sizeof(unsigned char));
            }
            pthread_testcancel();
        }


        // Get stats from packet socket
		hashpipe_pktsock_stats(p_ps, &pktsock_pkts, &pktsock_drops);

        //int packet_size = PKT_UDP_SIZE(p_frame) - 8;
        //printf("packet size is: %d\n", packet_size);
        //printf("for loop npackets: %lu\n", npackets);

        hashpipe_status_lock_safe(&st);
		hputi8(st.buf, "NPACKETS", npackets);
		//hputi8(st.buf,"PKTLOSS",pkt_loss);
		//hputr8(st.buf,"LOSSRATE",pkt_loss_rate);		
		hputu8(st.buf, "NETRECV",  pktsock_pkts);
		hputu8(st.buf, "NETDROPS", pktsock_drops);
		hashpipe_status_unlock_safe(&st);


        //Mark block as full
        if(HSD_input_databuf_set_filled(db, block_idx) != HASHPIPE_OK){
            hashpipe_error(__FUNCTION__, "error waiting for databuf filled call");
            pthread_exit(NULL);
        }

        db->block[block_idx].header.mcnt = mcnt;
		block_idx = (block_idx + 1) % db->header.n_block;
		mcnt++;

        //Will exit if thread has been cancelled
        pthread_testcancel();
    }

    pthread_cleanup_pop(1); /* Closes push(hashpipe_pktsock_close) */
	pthread_cleanup_pop(1); /* Closes push(free) */

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

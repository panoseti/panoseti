/*
 * HSD_net_thread.c
 * 
 * The net thread which is used to read packets from the quabos.
 * These packets are then written into the shared memory blocks,
 * which then allows for the pre-process of the data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include "hashpipe.h"
#include "HSD_databuf.h"

//PKTSOCK Params(These should be only changed with caution as it need to change with MMAP)
#define PKTSOCK_BYTES_PER_FRAME (16384)
#define PKTSOCK_FRAMES_PER_BLOCK (8)
#define PKTSOCK_NBLOCKS (20)
#define PKTSOCK_NFRAMES (PKTSOCK_FRAMES_PER_BLOCK * PKTSOCK_NBLOCKS)

//DEBUGGING MODE 
#define TEST_MODE

static int init(hashpipe_thread_args_t * args){

    // define default network params
    char bindhost[80];
    int bindport = 60001;
    hashpipe_status_t st = args->st;
    strcpy(bindhost, "0.0.0.0");

    //Locking shared buffer to properly get and set values.
    hashpipe_status_lock_safe(&st);

    // Get info from status buffer if present
    hgets(st.buf, "BINDHOST", 80, bindhost);
    hgeti4(st.buf, "BINDPORT", &bindport);

    //Store bind host/port info and other info in status buffer
    hputs(st.buf, "BINDHOST", bindhost);
	hputi4(st.buf, "BINDPORT", bindport);
    hputi8(st.buf, "NPACKETS", 0);

    //Unlocking shared buffer once complete.
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

    //Opening Pktsocket to recieve data.
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

//Struct to store the header vaules
typedef struct {
    int acqmode;
    uint16_t packetNum;
    uint16_t moduleNum;
    uint8_t quaboNum;
    uint32_t packet_UTC;
    uint32_t packet_NANOSEC;
    int boardLocation;
}packet_header_t;


/**
 * Check the acqmode of the packet coming in. Returns True if it is valid and returns
 * false if it is an acqmode that is not recognized
 * @param pkt_data The pointer for the packet frame(returned from PKT_UDP_DATA(p_frame))
 * @param pkt_header The header struct to be written to
 * @return 0 if acqmode is recognized and 1 otherwise
 */
int check_acqmode(unsigned char* pkt_data){
    return 1;
    printf("Checking the Acqmode %c", pkt_data[0]);
    if (pkt_data[0] == 1 || pkt_data[0] == 2 || pkt_data[0] == 3 ||
        pkt_data[0] == 6 || pkt_data[0] == 7){
            return 0;
        }

    /*
    printf("Malformed packet identify");
    for(int i=0; i < PKTDATASIZE; i++){
        if(i % 16 == 0){
            printf("\n");
        }
        printf("%02X ", pkt_data[i]);
    }*/
    return 1;
}

/**
 * Get the header info of the first packet in the PKTSOCK buffer
 * @param p_frame The pointer for the packet frame(returned from PKT_UDP_DATA(p_frame))
 * @param pkt_header The header struct to be written to
 */
static inline void get_header(unsigned char* pkt_data, int i, HSD_input_block_header_t* pkt_header) {
    /*uint64_t raw_header;
    raw_header = *(unsigned long long *)(PKT_UDP_DATA(p_frame));
    pkt_header->mode = (raw_header & 0x00000000000000ff);
    pkt_header->packetNum = (raw_header & 0x00000000ffff0000) >> (16);
    pkt_header->boardLocation = (raw_header & 0x0000ffff00000000) >> (32);*/
    pkt_header->acqmode[i] = pkt_data[0];
    pkt_header->pktNum[i] = ((pkt_data[3] << 8) & 0xff00) 
                        | (pkt_data[2] & 0x00ff);
    pkt_header->modNum[i] = ((pkt_data[5] << 6) & 0x3fc0) 
                        | ((pkt_data[4] >> 2) & 0x003f);
    pkt_header->quaNum[i] = ((pkt_data[4]) & 0x03);
    pkt_header->pktUTC[i] = ((pkt_data[9] << 24) & 0xff000000) 
                        | ((pkt_data[8] << 16) & 0x00ff0000)
                        | ((pkt_data[7] << 8) & 0x0000ff00)
                        | ((pkt_data[6]) & 0x000000ff);
                        
    pkt_header->pktNSEC[i] = ((pkt_data[13] << 24) & 0xff000000) 
                        | ((pkt_data[12] << 16) & 0x00ff0000)
                        | ((pkt_data[11] << 8) & 0x0000ff00)
                        | ((pkt_data[10]) & 0x000000ff);
    

    #ifdef TEST_MODE
        //printf("Mode:%i ", pkt_header->mode);
        //printf("BoardLocation:%02X ", pkt_header->boardLocation);
        //printf("Packet:%lu\n", pkt_header->packetNum);
        //printf("Header:%02lx\n", (raw_header & 0xffffffffffffffff));
        printf("Header");
        for (int i = 0; i < HEADERSIZE; i++){
            printf(" %X ", pkt_data[i]);
        }
        printf("\n");
    #endif
}

static int INTSIG;

void INThandler(int signum) {
    INTSIG = 1;
}

static void *run(hashpipe_thread_args_t * args){
    signal(SIGINT, INThandler);
    INTSIG = 0;
    
    //Creating pointers hashpipe args
    HSD_input_databuf_t *db  = (HSD_input_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    /* Main loop */

    //Variables 
    int rv, n;
    uint64_t mcnt = 0;          //Mcount of
    int block_idx = 0;          //The input buffer block index
    HSD_input_block_header_t* blockHeader;
    packet_header_t pkt_header; //Current packet's header
    unsigned char* pkt_data;    //Packet Data from PKT_UDP_DATA
    struct timeval nowTime;     //Current NTP UTC time
    int rc;                     
            
    //Input elements(Packets from Quabo)
    char *str_rcv, *str_q;
    str_rcv = (char *)malloc(PKTSIZE*sizeof(char));
    str_q = (char *)malloc(PKTSIZE*sizeof(char));

    //Compute the pkt_loss in the compute thread
    unsigned int pktsock_pkts = 0;      // Stats counter for socket packet
    unsigned int pktsock_drops = 0;     // Stats counter for dropped socket packet
    uint64_t npackets = 0;              // number of received packets
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

    printf("-----------Finished Setup of Input Thread------------\n\n");

    /* Main Loop */
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

        blockHeader = &(db->block[block_idx].header);
        blockHeader->data_block_size = 0;
        blockHeader->INTSIG = INTSIG;
        // Loop through all of the packets in the buffer block.
        for (int i = 0; i < N_PKT_PER_BLOCK; i++){
            //Check if the INTSIG is recognized
            if(INTSIG) break;
            //Recv all of the UDP packets from PKTSOCK
            do {
                do {
                    p_frame = hashpipe_pktsock_recv_udp_frame_nonblock(p_ps, bindport);
                } while (!p_frame && run_threads() && !INTSIG);

                if (INTSIG){break;}
            } while(!check_acqmode(PKT_UDP_DATA(p_frame)));

            //Check to see if the threads are still running. If not then terminate
            if(!run_threads() || INTSIG) break;

            //TODO
            //Check Packet Number at the beginning and end to see if we lost any packets
            npackets++;
            pkt_data = (unsigned char *) PKT_UDP_DATA(p_frame);
            get_header(pkt_data, i, blockHeader);
            
            #ifdef TEST_MODE
                //printf("PKT_UDP_DATA %i\n", PKT_UDP_DATA(p_frame)-8);
            #endif

            hashpipe_pktsock_release_frame(p_frame);

            //Copy the packets in PKTSOCK to the input circular buffer
            //Size is based on whether or not the mode is 16 bit or 8 bit
            if (blockHeader->acqmode[i] < 4){
                memcpy(db->block[block_idx].data_block+i*PKTDATASIZE, PKT_UDP_DATA(p_frame), PKTDATASIZE*sizeof(unsigned char));
            } else {
                memcpy(db->block[block_idx].data_block+i*PKTDATASIZE, PKT_UDP_DATA(p_frame), BIT8PKTDATASIZE*sizeof(unsigned char));
            }

            //Time stamping the packets and passing it into the shared buffer
            rc = gettimeofday(&nowTime, NULL);
            if (rc == 0){
                blockHeader->tv_sec[i] = nowTime.tv_sec;
                blockHeader->tv_usec[i] = nowTime.tv_usec;
            } else {
                printf("gettimeofday() failed, errno = %d\n", errno);
                blockHeader->tv_sec[i] = 0;
                blockHeader->tv_usec[i] = 0;
            }

            blockHeader->data_block_size++;

            #ifdef TEST_MODE
                //printf("TIME %li.%li\n", nowTime.tv_sec, nowTime.tv_usec);
            #endif
            pthread_testcancel();
        }


        // Get stats from packet socket
		hashpipe_pktsock_stats(p_ps, &pktsock_pkts, &pktsock_drops);

        #ifdef TEST_MODE
            //int packet_size = PKT_UDP_SIZE(p_frame) - 8;
            //printf("packet size is: %d\n", packet_size);
            //printf("for loop npackets: %lu\n", npackets);
        #endif

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

/**
 * Sets the functions and buffers for this thread
 */
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

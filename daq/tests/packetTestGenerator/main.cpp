#include <iostream>
#include <fstream>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <strings.h>

#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#define PKTSIZE16BIT 528
#define PKTSIZE8BIT 272

using namespace std;

void delay (unsigned int msecs) {
	clock_t goal = msecs*CLOCKS_PER_SEC/1000 + clock();  //convert msecs to clock count  
	while ( goal > clock() );               // Loop until it arrives.
}

uint8_t findPktNum(char data){
    return ((data << 4) & 0xf0) | ((data >> 4) & 0x0f);
}

uint8_t char_to_uint8(char ch){
    switch (ch){
        case '0': return 0x00;
        case '1': return 0x01;
        case '2': return 0x02;
        case '3': return 0x03;
        case '4': return 0x04;
        case '5': return 0x05;
        case '6': return 0x06;
        case '7': return 0x07;
        case '8': return 0x08;
        case '9': return 0x09;
        case 'a': return 0x0a;
        case 'b': return 0x0b;
        case 'c': return 0x0c;
        case 'd': return 0x0d;
        case 'e': return 0x0e;
        case 'f': return 0x0f;
        default: return 0x00;
    }
}

uint8_t hex_to_uint8(char first, char second){
    uint8_t firstu = (char_to_uint8(first)<<4);
    uint8_t secondu = char_to_uint8(second);
    return (char_to_uint8(first)<<4) + char_to_uint8(second);
}


vector<uint8_t>* hex_to_uint8(vector<char> *data){
    vector<uint8_t> *result = new vector<uint8_t>();

    for(auto i = 0; i < data->size(); i=i+2){
        result->push_back(hex_to_uint8((*data)[i], (*data)[i+1]));
    }

    return result;
}

int main(int argc, char** argv) {
    char *IPADDRESS = "127.0.0.1";
    uint32_t PORT = 60001;
    unsigned long frequency = 500;
    unsigned int module1 = 254;
    unsigned int module2 = 1;


    if (argc <= 1){
        cout << "A Packet Generator of Unit Tests for High speed data transport in PANOSETI. " << endl
             << endl
             << "The software was created at UC Berkeley by Ryan Lee under Dan Werthimer." << endl
             << endl
             << "Use the flag --help for more information" << endl;
        exit(0);
    }

    if (!strncmp(argv[1], "--help", 6)) {
        cout << "Program takes a sample input file to construct the packet, and also parameters to send the packet." << endl
             << "Packets will be send via UDP." << endl
             << endl << endl
             << "Flags: (Default values will be given to avoid error)" << endl
             << "-i : The IP address to send the packets to(Default:127.0.0.1)" << endl
             << "-p : The port which the packets will be sent(Default:60001)" << endl
             << "-n : The number of packets that will be sent(Default:1)" << endl
             << "-t : The desired frequency in which the 8 packet group should be sent (in Hz)(Default:500Hz). A zero value will send it as fast as possible" << endl
             << "-m : The two modules that will be used for testing (Default:254 and 1)" << endl
             << "     Note: that the program will attempt to run at this frequency but actual frequency will be displayed." << endl;
        exit(0);
    }

    /*if (argc%2 != 1) {
        cout << "ERROR: Incorrect number of arguements." << endl;
        exit(0);
    }*/

    for (int i = 1; i < argc; i=i+2) {

        if (!strncmp(argv[i], "-i", 2)) {
            IPADDRESS = argv[i+1];
        } else if (!strncmp(argv[i], "-p", 2)) {
            PORT = (uint32_t) stoul(argv[i+1]);
        } else if (!strncmp(argv[i], "-t", 2)) {
            frequency = stoul(argv[i+1]);
        } else if (!strncmp(argv[i], "-m", 2)) {
            module1 = stoul(argv[i+1]);
            module2 = stoul(argv[i+2]);
        }
    }
    module1 = module1 << 2;
    module2 = module2 << 2;

    unsigned int quabos[8] = {module1, module1+1, module1+2, module1+3,
                            module2, module2+1, module2+2, module2+3};
    unsigned char quabosHex[16];
    for (int i = 0; i < 8; i++){
        quabosHex[i*2] = quabos[i] & 0x00ff;
        quabosHex[i*2 + 1] = (quabos[i] >> 8) & 0x00ff;
    }
    
    int protocalValue = 0;
    long double Time = 0;
    if (frequency > 0){
        Time = 1/(double)frequency;
    }
    useconds_t sleepTime;
    if (Time < 0.000001)
        sleepTime = 0;
    else
        sleepTime = Time * 1000; 

    unsigned long successPackets = 0;
    
    // Setting up Socket to send message.

    struct sockaddr_in serv_addr;

    int sockfd = socket(AF_INET, SOCK_DGRAM, protocalValue);
    if (sockfd < 0){
        perror("Cannot Open Socket");
        exit(1);
    }


    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    serv_addr.sin_addr.s_addr = inet_addr(IPADDRESS);

    socklen_t addrLength = sizeof(serv_addr);



    struct sockaddr_in HK_serv_addr;

    int HKsock = socket(AF_INET, SOCK_DGRAM, protocalValue);
    if (HKsock < 0){
        perror("Cannot Open HK Socket");
    }

    HK_serv_addr.sin_family = AF_INET;
    HK_serv_addr.sin_port = htons(60002);
    HK_serv_addr.sin_addr.s_addr = inet_addr(IPADDRESS);

    socklen_t HK_addrLength = sizeof(HK_serv_addr);

    unsigned char *HKPacket = (unsigned char *)malloc(64*sizeof(unsigned char));
    HKPacket[0] = 0x20;
    memset(HKPacket+1, 0, 64*sizeof(unsigned char));
    for (int q = 0; q < 8; q++){
        HKPacket[2] = quabosHex[q*2];
        HKPacket[3] = quabosHex[q*2 + 1];
        if(sendto(sockfd, HKPacket, 64, 0,(struct sockaddr *) &HK_serv_addr, HK_addrLength) == 64){
            successPackets++;
        }
    }

    clock_t startTime = clock();

    printf("Starting 16 Bit Test\n");
    unsigned char *dataBytes = (unsigned char *)malloc(PKTSIZE16BIT*sizeof(unsigned char));

    dataBytes[0] = 0x02;
    dataBytes[1] = 0x00;
    dataBytes[2] = 0x00;
    dataBytes[3] = 0x00;
    dataBytes[4] = quabos[0] & 0x00ff;
    dataBytes[5] = (quabos[0] >> 8) & 0x00ff;
    for(int i = 6; i < PKTSIZE16BIT; i++){
        dataBytes[i] = 0x00;
    }

    for (int q = 0; q < 8; q++){
        dataBytes[4] = quabosHex[q*2];
        dataBytes[5] = quabosHex[q*2+1];
        if (sendto(sockfd, dataBytes, PKTSIZE16BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE16BIT)
            successPackets++;
    }
    dataBytes[2]++;
    if (dataBytes[2] == 0x00)
        dataBytes[3]++;
    printf("Sending Packets for Checking Image Data Lower Bit\n");
    for (int i = 0; i < 256; i++) {

        dataBytes[i*2 + 16] = 0x01;

        for (int q = 0; q < 8; q++){
            dataBytes[4] = quabosHex[q*2];
            dataBytes[5] = quabosHex[q*2+1];
            if (sendto(sockfd, dataBytes, PKTSIZE16BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE16BIT)
                successPackets++;
        }
        dataBytes[2]++;
        if (dataBytes[2] == 0x00)
            dataBytes[3]++;

        delay(sleepTime);
    }
    
    printf("Sending Packets for Checking Image Data Upper Bit\n");
    for (int i = 0; i < 256; i++) {

        dataBytes[i*2 + 17] = 0x01;

        for (int q = 0; q < 8; q++){
            dataBytes[4] = quabosHex[q*2];
            dataBytes[5] = quabosHex[q*2+1];
            if (sendto(sockfd, dataBytes, PKTSIZE16BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE16BIT)
                successPackets++;
        }
        dataBytes[2]++;
        if (dataBytes[2] == 0x00)
            dataBytes[3]++;

        delay(sleepTime);
    }

    printf("Sending Packets for Checking Image Data Max Value\n");
    for (int i = 0; i < 256; i++) {

        dataBytes[i*2 + 16] = 0xFF;
        dataBytes[i*2 + 17] = 0xFF;

        for (int q = 0; q < 8; q++){
            dataBytes[4] = quabosHex[q*2];
            dataBytes[5] = quabosHex[q*2+1];
            if (sendto(sockfd, dataBytes, PKTSIZE16BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE16BIT)
                successPackets++;
        }
        dataBytes[2]++;
        if (dataBytes[2] == 0x00)
            dataBytes[3]++;

        delay(sleepTime);
    }



    memset(HKPacket+1, 1, 64*sizeof(unsigned char));
    for (int q = 0; q < 8; q++){
        HKPacket[2] = quabosHex[q*2];
        HKPacket[3] = quabosHex[q*2 + 1];
        if(sendto(sockfd, HKPacket, 64, 0,(struct sockaddr *) &HK_serv_addr, HK_addrLength) == 64){
            successPackets++;
        }
    }


    dataBytes[0] = 0x01;
    dataBytes[1] = 0x00;
    dataBytes[2] = 0x00;
    dataBytes[3] = 0x00;
    dataBytes[4] = quabos[0] & 0x00ff;
    dataBytes[5] = (quabos[0] >> 8) & 0x00ff;
    for(int i = 6; i < PKTSIZE16BIT; i++){
        dataBytes[i] = 0x00;
    }

    if (sendto(sockfd, dataBytes, PKTSIZE16BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE16BIT)
        successPackets++;
    dataBytes[2]++;
    if (dataBytes[2] == 0x00)
        dataBytes[3]++;
    printf("Sending Packets for PH Data Lower Bit\n");
    for (int i = 0; i < 256; i++) {

        dataBytes[i*2 + 16] = 0x01;

        if (sendto(sockfd, dataBytes, PKTSIZE16BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE16BIT)
            successPackets++;
        dataBytes[2]++;
        if (dataBytes[2] == 0x00)
            dataBytes[3]++;

        delay(sleepTime);
    }
    printf("Sending Packets for PH Data Upper Bit\n");
    for (int i = 0; i < 256; i++) {

        dataBytes[i*2 + 17] = 0x01;

        if (sendto(sockfd, dataBytes, PKTSIZE16BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE16BIT)
            successPackets++;
        dataBytes[2]++;
        if (dataBytes[2] == 0x00)
            dataBytes[3]++;

        delay(sleepTime);
    }
    printf("Sending Packets for PH Data MAX Value\n");
    for (int i = 0; i < 256; i++) {

        dataBytes[i*2 + 16] = 0xFF;
        dataBytes[i*2 + 17] = 0xFF;

        if (sendto(sockfd, dataBytes, PKTSIZE16BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE16BIT)
            successPackets++;
        dataBytes[2]++;
        if (dataBytes[2] == 0x00)
            dataBytes[3]++;

        delay(sleepTime);
    }
    printf("Done\n\n");


    memset(HKPacket+1, 2, 64*sizeof(unsigned char));
    for (int q = 0; q < 8; q++){
        HKPacket[2] = quabosHex[q*2];
        HKPacket[3] = quabosHex[q*2 + 1];
        if(sendto(sockfd, HKPacket, 64, 0,(struct sockaddr *) &HK_serv_addr, HK_addrLength) == 64){
            successPackets++;
        }
    }


    printf("Starting 8 Bit Test\n");
    free(dataBytes);
    dataBytes = (unsigned char *)malloc(PKTSIZE8BIT*sizeof(unsigned char));

    dataBytes[0] = 0x06;
    dataBytes[1] = 0x00;
    dataBytes[2] = 0x00;
    dataBytes[3] = 0x00;
    dataBytes[4] = quabos[0] & 0x00ff;
    dataBytes[5] = (quabos[0] >> 8) & 0x00ff;
    for(int i = 6; i < PKTSIZE8BIT; i++){
        dataBytes[i] = 0x00;
    }
    for (int q = 0; q < 8; q++){
        dataBytes[4] = quabosHex[q*2];
        dataBytes[5] = quabosHex[q*2+1];
        if (sendto(sockfd, dataBytes, PKTSIZE8BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE8BIT)
            successPackets++;
    }
    dataBytes[2]++;
    if (dataBytes[2] == 0x00)
        dataBytes[3]++;
    printf("Sending Packets for Checking Image Data Lower Bit\n");
    for (int i = 0; i < 256; i++) {

        dataBytes[i + 16] = 0x01;

        for (int q = 0; q < 8; q++){
            dataBytes[4] = quabosHex[q*2];
            dataBytes[5] = quabosHex[q*2+1];
            if (sendto(sockfd, dataBytes, PKTSIZE8BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE8BIT)
                successPackets++;
        }
        dataBytes[2]++;
        if (dataBytes[2] == 0x00)
            dataBytes[3]++;

        delay(sleepTime);
    }

    memset(HKPacket+1, 3, 64*sizeof(unsigned char));
    for (int q = 0; q < 8; q++){
        HKPacket[2] = quabosHex[q*2];
        HKPacket[3] = quabosHex[q*2 + 1];
        if(sendto(sockfd, HKPacket, 64, 0,(struct sockaddr *) &HK_serv_addr, HK_addrLength) == 64){
            successPackets++;
        }
    }

    printf("Sending Packets for Checking Image Data Max Value\n");
    for (int i = 0; i < 256; i++) {

        dataBytes[i + 16] = 0xFF;

        for (int q = 0; q < 8; q++){
            dataBytes[4] = quabosHex[q*2];
            dataBytes[5] = quabosHex[q*2+1];
            if (sendto(sockfd, dataBytes, PKTSIZE8BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE8BIT)
                successPackets++;
        }
        dataBytes[2]++;
        if (dataBytes[2] == 0x00)
            dataBytes[3]++;

        delay(sleepTime);
    }
    printf("Done\n\n");

    memset(HKPacket+1, 0, 64*sizeof(unsigned char));
    HKPacket[36] = 0xFF;
    HKPacket[37] = 0xFF;
    for (int q = 0; q < 8; q++){
        HKPacket[2] = quabosHex[q*2];
        HKPacket[3] = quabosHex[q*2 + 1];
        if(sendto(sockfd, HKPacket, 64, 0,(struct sockaddr *) &HK_serv_addr, HK_addrLength) == 64){
            successPackets++;
        }
    }

    /*char answer = '\0';
    printf("Test Packets have been sent. Now please press Ctl+c on Hashpipe and type y to proceed:");
    scanf("%c", &answer);
    while (answer != 'y'){
        printf("Invalid Response. Please type y to proceed: ");
        scanf("%c", &answer);
    }*/

    dataBytes[0] = 0x06;
    dataBytes[1] = 0x00;
    dataBytes[2] = 0x00;
    dataBytes[3] = 0x00;
    dataBytes[4] = quabos[0] & 0x00ff;
    dataBytes[5] = (quabos[0] >> 8) & 0x00ff;
    for(int i = 6; i < PKTSIZE8BIT; i++){
        dataBytes[i] = 0x00;
    }

    for (int i = 0; i < 16; i++){
        if (sendto(sockfd, dataBytes, PKTSIZE8BIT, 0, (struct sockaddr *) &serv_addr, addrLength) == PKTSIZE8BIT)
            successPackets++;
    }



    printf("\n Finished \n");

    //float totalTime = (float)(clock()-startTime);

    //totalTime = totalTime/CLOCKS_PER_SEC;

    close(HKsock);
    close(sockfd);

    //cout << "Average Frequency of 8 Packet Groups" << to_string((int)(successPackets/(totalTime*8))) << "Hz" << endl
    //     << to_string(successPackets) << " Successful Packets Sent out of " << to_string(successPackets) << " Packets" << endl;

    free(dataBytes);

    return 0;
}

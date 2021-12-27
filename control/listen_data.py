#! /usr/bin/env python

# listen for data packets (from any quabo)
# show their type and source

import socket, time

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', 60001))
    while True:
        x = sock.recvfrom(2048)
        # returns [data, [ip_addr, port]]
        data = x[0]
        ip_addr = [1][0]
        print('got %d bytes from %s'%(len(data), ip_addr))

main()
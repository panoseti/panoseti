#! /usr/bin/env python

# listen for housekeeping packets (from any quabo)
# show their type and source

import socket, time

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', 60002))
    while True:
        x = sock.recvfrom(2048)
        # returns [data, [ip_addr, port]]
        data = x[0]
        ip_addr = x[1][0]
        print('%f: got %d bytes from %s'%(time.time(), len(data), ip_addr))

main()

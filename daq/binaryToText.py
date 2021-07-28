def hex_to_char(val):
    switcher = {
        0x00: '0',
        0x01: '1',
        0x02: '2',
        0x03: '3',
        0x04: '4',
        0x05: '5',
        0x06: '6',
        0x07: '7',
        0x08: '8',
        0x09: '9',
        0x0a: 'a',
        0x0b: 'b',
        0x0c: 'c',
        0x0d: 'd',
        0x0e: 'e',
        0x0f: 'f',
    }
    return switcher.get(val, "=")

binary = open("data.out", "rb")
byte = binary.read()
print("File Length in Bytes")
print(len(byte))
binary.close()

out = open("data.txt", "a")
index = 0
PKTNUM = 0
PKTSIZE = 528
out.write("----------------------------\n")
out.write("Packet 0\n")
for i in byte:
    out.write(hex_to_char((i >> 4) & 0x0f))
    out.write(hex_to_char(i & 0x0f))
    if index % PKTSIZE == PKTSIZE-1:
        PKTNUM += 1
        out.write("\n")
        out.write("----------------------------\n")
        out.write("Packet ")
        out.write(str(PKTNUM))
        out.write('\n')
    else:
        out.write(' ')
    index += 1
out.close()

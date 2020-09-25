import socket
import time
HOST = '127.0.0.1'
PORT = 60002
NPACKETS = 1

PACKET = [
    512,                        # BOARDLOC
    -40, -41, -42, -43,          # HVMON (0 to -80V)
    32767, 32768, 32769, 32770, # HVIMON ((65535-HVIMON) * 38.1nA) (0 to 2.5mA)
    -44,                        # RAWHVMON (0 to -80V)
    32771,                      # V12MON (19.07uV/LSB) (1.2V supply)
    32772,                      # V18MON (19.07uV/LSB) (1.8V supply)
    32773,                      # V33MON (38.10uV/LSB) (3.3V supply)
    32774,                      # V37MON (38.10uV/LSB) (3.7V supply)
    32775,                      # I10MON (182uA/LSB) (1.0V supply)
    32776,                      # I18MON (37.8uA/LSB) (1.8V supply)
    32777,                      # I33MON (37.8uA/LSB) (3.3V supply)
    320,                        # TEMP1 (0.0625*N)
    38121,                      # TEMP2 (N/130.04-273.15)
    32778,                      # VCCINT (N*3/65536)
    32779,                      # VCCAUX (N*3/65536)
    1,0,0,0,                    # UID
    1,                          # SHUTTER and LIGHT_SENSOR STATUS
    0,                          # unused
    0,1,1,0                     # FWID0 and FWID1
]

signed = [
    0,                        # BOARDLOC
    1, 1, 1, 1,          # HVMON (0 to -80V)
    0, 0, 0, 0, # HVIMON ((65535-HVIMON) * 38.1nA) (0 to 2.5mA)
    1,                        # RAWHVMON (0 to -80V)
    0,                      # V12MON (19.07uV/LSB) (1.2V supply)
    0,                      # V18MON (19.07uV/LSB) (1.8V supply)
    0,                      # V33MON (38.10uV/LSB) (3.3V supply)
    0,                      # V37MON (38.10uV/LSB) (3.7V supply)
    0,                      # I10MON (182uA/LSB) (1.0V supply)
    0,                      # I18MON (37.8uA/LSB) (1.8V supply)
    0,                      # I33MON (37.8uA/LSB) (3.3V supply)
    1,                        # TEMP1 (0.0625*N)
    0,                      # TEMP2 (N/130.04-273.15)
    0,                      # VCCINT (N*3/65536)
    0,                      # VCCAUX (N*3/65536)
    0,0,0,0,                    # UID
    0,                        # SHUTTER and LIGHT_SENSOR STATUS
    0,                        # unused
    0,0,0,0                     # FWID0 and FWID1
]

packet1 = b'\x20\xaa'

packet2 = b'\x20\x00'

packet = packet1


for i, sign in zip(PACKET, signed):
    packet += i.to_bytes(2, 'little', signed=sign)
    
def updatePacket(template):
    val = packet2
    for i, sign in zip(PACKET, signed):
        val += i.to_bytes(2, 'little', signed=sign)
    return val

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.sendto(packet, (HOST,PORT))
    
    for i in range(NPACKETS-1):
        time.sleep(4)
        packet = updatePacket(PACKET)
        sock.sendto(packet, (HOST,PORT))

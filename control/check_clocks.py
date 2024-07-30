#! /usr/bin/env python

# 'check_clocks' is a class for checking time synchronization in PANOSETI system.
# There are several functions in this class for getting time from GPS receiver, quabo and WRS.
# 
import time
import serial
from datetime import datetime, timezone
import socket
import paramiko
import struct
import qstart
import qstart, config_file

# this is the offset time between tai and utc
LEAP_SEC = 37
# this is the offset time between gps and utc
GPS_SEC  = 18
# time difference between host time and device time
TOLERANCE = 0.5

# create ssh session for getting wrs time
#
def SSH_Init(wrs_ip):
    ssh=paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(wrs_ip,username='root',password='')
    return ssh


# The class is used for checking timing synchronization in PANOSETI system.
#
class check_clocks(object):
    def __init__(self, gps_port='/dev/ttyUSB0', wrs_ip='192.168.1.254', host_ip='192.168.1.100', port=60001):
        self.gps_port = gps_port
        self.host_ip = host_ip
        self.port = port
        self.wrs_ip = wrs_ip
        self.ser = 0
        self.ssh =  0

    
    # parse primary timing packets from GPS receiver
    #
    def _parse_primary_packet(self,data):
        # check the length of data
        if len(data) != 17:
            return
        BYTEORDER = 'big'
        # get time info from the data packet
        seconds = int.from_bytes(data[10:11], byteorder=BYTEORDER, signed=False)
        minutes = int.from_bytes(data[11:12], byteorder=BYTEORDER, signed=False)
        hours = int.from_bytes(data[12:13], byteorder=BYTEORDER, signed=False)
        dayofMonth = int.from_bytes(data[13:14], byteorder=BYTEORDER, signed=False)
        month = int.from_bytes(data[14:15], byteorder=BYTEORDER, signed=False)
        year = int.from_bytes(data[15:17], byteorder=BYTEORDER, signed=False)
        # there is no nanosec info from GPS receiver, so nanosec value is set to 0 here
        lastTime = datetime(year, month, dayofMonth, hours, minutes, seconds, 0).replace(tzinfo=timezone.utc)
        return lastTime.timestamp() - GPS_SEC
 
    
    # Get time from GPS receiver.
    # The information is from a serial port, which is defined in obs_config.json.
    #
    def get_gps_time(self):

        self.ser = serial.Serial(
            port=self.gps_port,
            baudrate=9600,
            timeout=1,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )

        if not self.ser.isOpen():
            self.ser.open()

        data = b''
        dataSize = 0
        bytesToRead = 0
        timestamp = False
        gps_time = []
        recv_byte = 0
        last_recv_byte = 0
        recv_state = True

        while(recv_state):
            # get gps packets from uart port
            while bytesToRead == 0:
                bytesToRead = self.ser.inWaiting()
            recv_byte = self.ser.read(bytesToRead)
            if(recv_byte == b'\x10' and last_recv_byte == b'\x10'):
                pass
            else:
                if(timestamp == False):
                    t_host = time.time()
                    timestamp = True
                data += recv_byte
                dataSize += bytesToRead
            last_recv_byte = recv_byte
            bytesToRead = 0

            # deal with the data packet, if the packet ends with \x10\x03
            if data[dataSize-1:dataSize] == b'\x03' and data[dataSize-2:dataSize-1] == b'\x10':
                if data[0:1] == b'\x10':
                    id = data[1:3]
                    if id == b'\x8f\xab':
                        gps_time = self._parse_primary_packet(data[2:dataSize-2])
                        if(gps_time !=None):
                            recv_state = False
                data = b''
                dataSize = 0
                timestamp = False

        self.ser.close()
        return gps_time, t_host

    
    # Get time from a quabo.
    # We just use 1 quabo for the test, and the IP address of the quabo is defined in obs_config.json.
    #
    def get_quabo_time(self):
        BUFFERSIZE = 1024
        IP_PORT = (self.host_ip,self.port)
        server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        server.bind(IP_PORT)
        server.settimeout(1)
        try:
            data,client_addr = server.recvfrom(BUFFERSIZE)
        except:
            raise Exception('\n No packets from Quabo!\n Please make sure the quabo is powered on and rebooted.')
        server.close()

        t_host = time.time()
        nanosec = struct.unpack("<I", data[10:14])[0]
        wr_tai = struct.unpack("<I", data[6:10])[0]
        wr_tai_10bits = wr_tai & 0x3ff
        #covert utc to tai
        host_tai = time.time() + LEAP_SEC
        #covert tai back to utc
        t_quabo = (int(host_tai) & 0xFFFFFFFFFFFFFC00) + wr_tai_10bits + nanosec/1000000000 - LEAP_SEC
        
        return t_quabo, t_host

    
    # Get time from WRS.
    # The information is from a ssh session, which requires the IP address of the WRS.
    # The IP address is defined in obs_config.json.
    #
    def get_wrs_time(self):
        cmd0 = "/wr/bin/wr_date get"

        self.ssh =  SSH_Init(self.wrs_ip)
        ssh_stdin, ssh_stdout, ssh_stderr = self.ssh.exec_command(cmd0)
        r0=ssh_stdout.read()
        t_host = time.time()
        
        r0_str=str(r0, encoding = "utf-8")
        s=r0_str.split(' ')
        wrs_time = float(s[0]) - LEAP_SEC
        self.ssh.close()
        del(self.ssh,ssh_stdin, ssh_stdout, ssh_stderr)
        return wrs_time, t_host

    
    # Compare the GPS time with host computer time.
    #
    def check_gps_time(self):
        t_gps, t_host = self.get_gps_time()
        if(abs(t_gps - t_host) < TOLERANCE):
            return True
        else:
            return False

    
    # Compare the quabo time with host computer time.
    #  
    def check_quabo_time(self):
        t_quabo, t_host = self.get_quabo_time()
        if(abs(t_quabo - t_host) < TOLERANCE):
            return True
        else:
            return False
    
    
    # Compare the WRS time with host computer time.
    #
    def check_wrs_time(self):
        t_wrs, t_host = self.get_wrs_time()
        if(abs(t_wrs - t_host) < TOLERANCE):
            return True
        else:
            return False
    
    
    # Check the time from GPS reciever, quabo and WRS.
    # If all of the time is good, it means the system timing is synced.
    #
    def check_time_sync(self):
        s0 = self.check_gps_time()
        s1 = self.check_quabo_time()
        s2 = self.check_wrs_time()
        if(s0 and s1 and s2):
            return True
        else:
            return False

if __name__ == '__main__':
    # get uart_port, wrs_ip from config file
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    gps_port = obs_config['gps_port']
    wrs_ip = socket.gethostbyname(obs_config['wr_ip_addr'])
    host_ip = socket.gethostbyname(daq_config['head_node_ip_addr'])

    print('===============================================================')
    print('Please make sure:')
    print('1. The dev name of the GPS receiver is'.ljust(46,' '),gps_port)
    print('2. The IP address of the host computer is'.ljust(46,' '),host_ip)
    print('3. The IP address of WRS is'.ljust(46,' '),wrs_ip)
    print('===============================================================')
    print('Time Checking Result(UTC TIME):')
    
    cts = check_clocks(gps_port, wrs_ip, host_ip)
    
    # get gps time from gps reciever
    t_gps,t_host = cts.get_gps_time()
    r0 = cts.check_gps_time()
    
    # start image mode on 1 quabo
    qstart.qstart(True)
    t_quabo = 0
    t_host1 = 0
    r1 = True
    # get quabo time
    t_quabo, t_host1 = cts.get_quabo_time()
    r1 = cts.check_quabo_time()
    # stop image mode on the quabo
    qstart.qstart(False)

    # get wrs time from white rabbit switch
    t_wrs, t_host2 = cts.get_wrs_time()
    r2 = cts.check_wrs_time()

    print('GPS Time'.ljust(20, ' '),':',t_gps)
    print('GPS Timestamp'.ljust(20,' '),':',t_host)
    print('Checking Result'.ljust(20,' '),':',r0,'\n')
    print('Quabo Time'.ljust(20,' '),':',t_quabo)
    print('Quabo Timestamp'.ljust(20,' '),':',t_host1)
    print('Checking Result'.ljust(20,' '),':',r1,'\n')
    print('WRS Time'.ljust(20, ' '),':',t_wrs)
    print('WRS Timestamp'.ljust(20,' '),':',t_host2)
    print('Checking Result'.ljust(20,' '),':',r2,'\n')

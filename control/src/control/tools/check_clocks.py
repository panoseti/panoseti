#! /usr/bin/env python

# 'check_clocks' is a class for checking time synchronization in PANOSETI system.
# There are several functions in this class for getting time from GPS receiver, quabo and WRS.
# 


import socket
import struct
import time
from datetime import UTC, datetime
from typing import Literal

import paramiko
import serial
from check_clocks import qstart

from control.utils import config_file

# this is the offset time between tai and utc
LEAP_SEC = 37
# this is the offset time between gps and utc
GPS_SEC  = 18
# time difference between host time and device time
TOLERANCE = 0.5

# create ssh session for getting wrs time
#
def SSH_Init(wrs_ip: str) -> paramiko.SSHClient:
    """Initialize an SSH session for communication with the White Rabbit Switch.

    Args:
        wrs_ip: IP address of the White Rabbit Switch.

    Returns:
        An active paramiko.SSHClient session.
    """
    ssh=paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(wrs_ip,username='root',password='')
    return ssh


class check_clocks:
    """Utilities for verifying time synchronization across the PANOSETI observatory.
    
    Provides methods to retrieve and compare UTC timestamps from a GPS receiver, 
    Quabo FPGA hardware, and the White Rabbit Switch (WRS).
    """

    def __init__(self, gps_port: str = '/dev/ttyUSB0', wrs_ip: str = '192.168.1.254', host_ip: str = '192.168.1.100', port: int = 60001) -> None:
        """Initialize the clock checker with device connectivity details.

        Args:
            gps_port: TTY path for the GPS serial interface.
            wrs_ip: IP address of the White Rabbit Switch.
            host_ip: IP address of the local head node.
            port: UDP port used for Quabo HK/data packets.
        """
        self.gps_port = gps_port
        self.host_ip = host_ip
        self.port = port
        self.wrs_ip = wrs_ip
        self.ser: serial.Serial | int = 0
        self.ssh: paramiko.SSHClient | int =  0

    
    def _parse_primary_packet(self, data: bytes) -> float | None:
        """Parse a primary timing packet (binary) from the GPS receiver.

        Args:
            data: Raw bytes from the serial interface.

        Returns:
            The parsed GPS timestamp in seconds (Unix epoch), adjusted to UTC.
        """
        # check the length of data
        if len(data) != 17:
            return None
        BYTEORDER: Literal['big', 'little'] = 'big'
        # get time info from the data packet
        seconds = int.from_bytes(data[10:11], byteorder=BYTEORDER, signed=False)
        minutes = int.from_bytes(data[11:12], byteorder=BYTEORDER, signed=False)
        hours = int.from_bytes(data[12:13], byteorder=BYTEORDER, signed=False)
        dayofMonth = int.from_bytes(data[13:14], byteorder=BYTEORDER, signed=False)
        month = int.from_bytes(data[14:15], byteorder=BYTEORDER, signed=False)
        year = int.from_bytes(data[15:17], byteorder=BYTEORDER, signed=False)
        # there is no nanosec info from GPS receiver, so nanosec value is set to 0 here
        lastTime = datetime(year, month, dayofMonth, hours, minutes, seconds, 0).replace(tzinfo=UTC)
        return lastTime.timestamp() - GPS_SEC
 
    
    def get_gps_time(self) -> tuple[float | None, float]:
        """Retrieve current time from the GPS receiver serial port.

        Returns:
            A tuple of (gps_utc_seconds, local_host_seconds).
        """

        self.ser = serial.Serial(
            port=self.gps_port,
            baudrate=9600,
            timeout=1,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )

        if isinstance(self.ser, serial.Serial) and not self.ser.isOpen():
            self.ser.open()

        data = b''
        dataSize = 0
        bytesToRead = 0
        timestamp = False
        gps_time: float | None = None
        t_host: float = 0.0
        recv_byte = b''
        last_recv_byte = b''
        recv_state = True

        while(recv_state):
            # get gps packets from uart port
            while bytesToRead == 0:
                if isinstance(self.ser, serial.Serial):
                    bytesToRead = self.ser.inWaiting()
            if isinstance(self.ser, serial.Serial):
                recv_byte = self.ser.read(bytesToRead)
            if(recv_byte == b'\x10' and last_recv_byte == b'\x10'):
                pass
            else:
                if(not timestamp):
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
                        if(gps_time is not None):
                            recv_state = False
                data = b''
                dataSize = 0
                timestamp = False

        if isinstance(self.ser, serial.Serial):
            self.ser.close()
        return gps_time, t_host

    
    def get_quabo_time(self) -> tuple[float, float]:
        """Sniff a UDP packet from a Quabo and decode its hardware timestamp.

        Returns:
            A tuple of (quabo_utc_seconds, local_host_seconds).

        Raises:
            Exception: If no packets are received from the Quabo within 1 second.
        """
        BUFFERSIZE = 1024
        IP_PORT = (self.host_ip,self.port)
        server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        server.bind(IP_PORT)
        server.settimeout(1)
        try:
            data, _client_addr = server.recvfrom(BUFFERSIZE)
        except Exception:
            raise Exception('\n No packets from Quabo!\n Please make sure the quabo is powered on and rebooted.') from None
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

    
    def get_wrs_time(self) -> tuple[float, float]:
        """Retrieve current time from the White Rabbit Switch via SSH.

        Returns:
            A tuple of (wrs_utc_seconds, local_host_seconds).
        """
        cmd0 = "/wr/bin/wr_date get"

        self.ssh =  SSH_Init(self.wrs_ip)
        _ssh_stdin, ssh_stdout, _ssh_stderr = self.ssh.exec_command(cmd0)
        r0=ssh_stdout.read()
        t_host = time.time()
        
        r0_str=str(r0, encoding = "utf-8")
        s=r0_str.split(' ')
        wrs_time = float(s[0]) - LEAP_SEC
        self.ssh.close()
        return wrs_time, t_host
    
    def check_gps_time(self) -> bool:
        """Verify GPS time is synchronized with the host clock.

        Returns:
            True if the time difference is within TOLERANCE.
        """
        t_gps, t_host = self.get_gps_time()
        return t_gps is not None and (abs(t_gps - t_host) < TOLERANCE)

    
    def check_quabo_time(self) -> bool:
        """Verify Quabo hardware time is synchronized with the host clock.

        Returns:
            True if the time difference is within TOLERANCE.
        """
        t_quabo, t_host = self.get_quabo_time()
        return abs(t_quabo - t_host) < TOLERANCE
    
    
    def check_wrs_time(self) -> bool:
        """Verify White Rabbit Switch time is synchronized with the host clock.

        Returns:
            True if the time difference is within TOLERANCE.
        """
        t_wrs, t_host = self.get_wrs_time()
        return abs(t_wrs - t_host) < TOLERANCE
    
    
    def check_time_sync(self) -> bool:
        """Verify that all observatory timing systems (GPS/Quabo/WRS) are synced.

        Returns:
            True if all systems pass synchronization checks.
        """
        s0 = self.check_gps_time()
        s1 = self.check_quabo_time()
        s2 = self.check_wrs_time()
        return bool(s0 and s1 and s2)

if __name__ == '__main__':
    # get uart_port, wrs_ip from config file
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    gps_port = str(obs_config.gps_port) if obs_config.gps_port else "/dev/ttyUSB0"
    wrs_ip = socket.gethostbyname(str(obs_config.wr_ip_addr))
    host_ip = socket.gethostbyname(str(daq_config.head_node_ip_addr))

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
    t_quabo: float = 0.0
    t_host1: float = 0.0
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

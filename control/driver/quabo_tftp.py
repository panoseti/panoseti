import logging
import os
import struct

import tftpy


class tftpw:
    def __init__(self, ip: str, port: int = 69) -> None:
        self.ip = ip
        self.client = tftpy.TftpClient(ip, port)
        if not os.path.exists('logs'):
            os.makedirs('logs')
        self.logger = logging.getLogger('PANOSETI.TFTPW')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('logs/quabo_tftp.log', mode='w')
        logformat = logging.Formatter('%(levelname)s - %(asctime)s - %(message)s')
        handler.setFormatter(logformat)
        self.logger.addHandler(handler)
        # deal with the tftpy warning messages
        log_tags = ["tftpy.TftpStates", "tftpy.TftpContext"]
        for tag in log_tags:
            logger = logging.getLogger(tag)
            logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler('logs/tftpy.log', mode='w')
            logformat = logging.Formatter('%(levelname)s - %(asctime)s - %(message)s')
            handler.setFormatter(logformat)
            if logger.handlers:
                logger.handlers.clear()
            logger.addHandler(handler)
    
    #print help information
    def help(self) -> None:
        print('Help Information:')
        print('  get_flashid()           : get flash id from flash chip, and the flash id are 8 bytes.')
        print('  get_wrpc_filesys()      : get wrpc file system from flash chip. [ default space : 0x00E00000--0x00F0FFFF]')
        print('  get_mb_file()           : get mb file from flash chip. [default space : 0x00F10000--0x0100FFFF]')
        print('  put_wrpc_filesys(file)  : put file to wrpc filesys space. [default space : 0x00E00000--0x00F0FFFF]')
        print('  put_mb_file(file)       : put file to mb file space. [default space : 0x00F10000--0x0100FFFF]')
        print('  put_bin_file(file)      : put fpga bin file to flash chip. [default from 0x01010000]')
        print('  reboot()                : reboot fpga. [default from 0x01010000]')
    
    #get flash_id
    def get_flashuid(self, filename: str = 'flashuid') -> None:
        self.logger.info(f'get_flashuid: filename - {filename}')
        self.client.download('/flashuid', filename, timeout=3)
        print('Get flash Device ID successfully!')
    
    #get wrpc_filesys
    #space     : 0x00E00000--0x00F0FFFF
    #size    : 1MB + 64K BYTES = 1114112 BYTES
    def get_wrpc_filesys(self, filename: str = 'wrpc_filesys', addr: int = 0x00e00000) -> None:
        self.logger.info(f'get_wrpc_filesys: filename - {filename}, addr - 0x{addr:08x}')
        fp_w = open(filename, 'wb')
        #we can get 65535 bytes each time, so we need to repeat the download operation for 16 times
        #for convenience, we read 32768 bytes each time
        for i in range(0, 34):
            addr_tmp = addr + i * 0x8000 
            offset = str(hex(addr_tmp))
            remote_filename = '/flash.' + offset[2:] + '.8000'
            #print('remote_filename :',remote_filename)
            #download the file to 'tmp'
            self.client.download(remote_filename, 'tmp')
            #open 'tmp'
            fp_r = open('tmp', 'rb')
            #read data out
            data = fp_r.read()
            #write the data to the final file
            fp_w.write(data)
            #close 'tmp'
            fp_r.close()
        fp_r.close()
        fp_w.close()
        os.remove('tmp')
        print('Download wrpc file system successfully!')
        
    #get mb_file 
    #space    : 0x00F10000--0x0100FFFF
    #size    : 1MB = 1048576 BYTES
    def get_mb_file(self, filename: str = 'mb_file', addr: int = 0x00F10000) -> None:
        self.logger.info(f'get_mb_file: filename - {filename}, addr - 0x{addr:08x}')
        fp_w = open(filename, 'wb')
        #we can get 65535 bytes each time, so we need to repeat the download operation for 16 times
        #for convenience, we read 32768 bytes each time
        for i in range(0, 32):
            addr_tmp = addr + i * 0x8000 
            offset = str(hex(addr_tmp))
            remote_filename = '/flash.' + offset[2:] + '.8000'
            #print('remote_filename :',remote_filename)
            #download the file to 'tmp'
            self.client.download(remote_filename, 'tmp')
            #open 'tmp'
            fp_r = open('tmp', 'rb')
            #read data out
            data = fp_r.read()
            #write the data to the final file
            fp_w.write(data)
            #close 'tmp'
            fp_r.close()
        fp_w.close()
        os.remove('tmp')
        print('Download mb file successfully!')
        
    #put wprc_filesys, starting from 0x00E00000
    def put_wrpc_filesys(self, filename: str = 'wrpc_filesys', addr: int = 0x00E00000) -> None:
        self.logger.info(f'put_wrpc_filesys: filename - {filename}, addr - 0x{addr:08x}')
        offset = str(hex(addr))
        remote_filename = '/flash.' + offset[2:]
        #print('remote_filename  ',remote_filename)
        size = os.path.getsize(filename)
        #check the size of wrpc_filesys
        if size != 0x110000 :
            print('The size of wrpc_filesys is incorrect, please check it!')
            return
        self.client.upload(remote_filename, filename)    
        # print('Upload %s to panoseti wrpc_filesys space successfully!' %filename)
        
    #put mb file, starting from 0x00F10000
    def put_mb_file(self, filename: str = 'mb_file', addr: int = 0x00F10000) -> None:
        self.logger.info(f'put_mb_file: filename - {filename}, addr - 0x{addr:08x}')
        offset = str(hex(addr))
        remote_filename = '/flash.' + offset[2:]
        #print('remote_filename  ',remote_filename)
        size = os.path.getsize(filename)
        #check the size of mb_file
        if size > 0x100000 :
            print('The size of mb file is too large, and it will mess up other parts on the flash chip!')
            return
        self.client.upload(remote_filename, filename)
        print(f'Upload {filename} to panoseti mb_file space successfully!')
        
    #put bin file,starting from 0x01010000
    def put_bin_file(self, filename: str, addr: int = 0x01010000) -> None:
        self.logger.info(f'put_bin_file: filename - {filename}, addr - 0x{addr:08x}')
        offset = str(hex(addr))
        remote_filename = '/flash.' + offset[2:]
        #print('remote_filename :',remote_filename)
        self.client.upload(remote_filename, filename)
        print(f'Upload {filename} to panoseti bin file space successfully!')
        
    def reboot(self, addr: int = 0x00010100) -> None:
        self.logger.info(f'reboot: addr - 0x{addr:08x}')
        remote_filename = '/progdev'
        filename = f'tmp.prog.{self.ip}'
        fp = open(filename, 'wb')
        for i in range(1, 5):
            s = struct.pack('B', addr>>(8*(4-i))&0xFF)
            fp.write(s)
        fp.close()
        # print('*******************************************************')
        # print('FPGA is rebooting, just ignore the timeout information')
        # print('Wait for 30s, and then check housekeeping data!')
        # print('*******************************************************')
        try:
            self.client.upload(remote_filename, filename)
        except Exception:
            pass
        os.remove(filename)
        

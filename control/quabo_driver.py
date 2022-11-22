# class for communicating with a quabo board
#
# example:
#   q = QUABO("128.5.4.2")
#   q.lf(False)     # set LED flasher 0
#
# Some of the operations get their info from a "quabo config file".
# Where possible I'd like to add variants that get params directly.
#
# A QUABO object has info about the quabo state,
# e.g. MAROC regs, shutter status etc.
# Currently these are initialized to zero.
# Ideally we should get the actual values from the quabo.
#
# See https://github.com/panoseti/panoseti/wiki/Quabo-device-driver

import socket, time
import util

UDP_CMD_PORT= 60000
    # port used on both sides for command packets
UDP_HK_PORT= 60002
    # used on master to receive HK packets

SERIAL_COMMAND_LENGTH = 829

# bits for data acquisition mode command
ACQ_PULSE_HEIGHT = 0x1
ACQ_IMAGE = 0x2
ACQ_IMAGE_8BIT = 0x4
ACQ_NO_BASELINE_SUBTRACT = 0x10

class DAQ_PARAMS:
    def __init__(self, do_image, image_us, image_8bit, do_ph, bl_subtract):
        self.do_image = do_image
        self.image_us = image_us
        self.image_8bit = image_8bit
        self.do_ph = do_ph
        self.bl_subtract = bl_subtract
        self.do_flash = False
    def set_flash_params(self, rate, level, width):
        self.do_flash = True
        self.flash_rate = rate
        self.flash_level = level
        self.flash_width = width

# currently each QUABO object has its own sockets,
# which means you can only have one at a time.
# will probably need to change this at some point

class QUABO:
    def __init__(self, ip_addr, config_file_path='quabo_config.txt'):
        self.ip_addr = ip_addr
        self.config_file_path = config_file_path
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        self.sock.bind(("", UDP_CMD_PORT))
        self.have_hk_sock = False

        self.shutter_open = 0
        self.shutter_power = 0
        self.fanspeed = 0
        self.HV_vals = [0,0,0,0]
        self.MAROC_regs = []
        for i in range (4):
            self.MAROC_regs.append([0 for x in range(104)])

    def close(self):
        self.sock.close()

    def send_daq_params(self, params):
        cmd = self.make_cmd(0x03)
        mode = 0

        if params.do_image:
            mode |= ACQ_IMAGE
        if params.image_8bit:
            mode |= ACQ_IMAGE_8BIT
        if params.do_ph:
            mode |= ACQ_PULSE_HEIGHT
        if not params.bl_subtract:
            mode |= ACQ_NO_BASELINE_SUBTRACT
        cmd[2] = mode
        cmd[4] = params.image_us % 256
        cmd[5] = params.image_us // 256
        cmd[12] = 70
        if params.do_flash:
            cmd[22] = params.flash_rate
            cmd[24] = params.flash_level
            cmd[26] = params.flash_width
        #util.print_binary(cmd)
        self.send(cmd)

    def send_maroc_params_file(self):
        cmd = bytearray(492)
        with open(self.config_file_path) as f:
            config = parse_quabo_config_file(self.config_file_path)
        self.make_maroc_cmd(config, cmd)
        self.flush_rx_buf()
        self.send(cmd)

    def send_maroc_params(self, config):
        cmd = bytearray(492)
        self.make_maroc_cmd(config, cmd)
        self.send(cmd)

    # returns the list of 256 coefficients
    #
    def calibrate_ph_baseline(self):
        cmd = self.make_cmd(0x07)
        self.flush_rx_buf()
        self.send(cmd)
        time.sleep(2)
        reply = self.sock.recvfrom(1024)
        bytesback = reply[0]
        x = []
        for n in range(256):
            val = bytesback[2*n+4] + 256*bytesback[2*n+5]
            x.append(val)
        return x

    def hv_config(self):
        cmd = self.make_cmd(0x02)
        with open(self.config_file_path) as f:
            self.parse_hv_params(f, cmd)
        self.flush_rx_buf()     # needed?
        self.send(cmd)

    def hv_set_chan(self, chan, value):
        cmd = self.make_cmd(0x02)
        self.HV_vals[chan] = int(value)
        for i in range(4):
            LSbyte = self.HV_vals[i] & 0xff
            MSbyte = (self.HV_vals[i] >> 8) & 0xff
            cmd[2*i+2]=LSbyte
            cmd[2*i+3]=MSbyte
        self.send(cmd)

    # set high voltage for all 4 channels
    #
    def hv_set(self, values):
        cmd = self.make_cmd(0x02)
        for i in range(4):
            cmd[2*i+2] = values[i] & 0xff
            cmd[2*i+3] = (values[i] >> 8) & 0xff
        self.send(cmd)

    def send_acq_parameters_file(self):
        cmd = self.make_cmd(0x03)
        with open(self.config_file_path) as f:
            self.parse_acq_parameters(f, cmd)
        self.flush_rx_buf()
        self.send(cmd)

    def send_trigger_mask(self):
        cmd = self.make_cmd(0x06)
        with open(self.config_file_path) as f:
            self.parse_trigger_mask(f, cmd)
        self.flush_rx_buf()
        self.send(cmd)

    def reset(self):
        cmd = self.make_cmd(0x04)
        self.send(cmd)

    def focus(self, steps):      # 1 to 50000, 0 to recalibrate
        endzone = 300
        backoff = 200
        step_ontime = 10000
        step_offtime = 10000

        cmd = self.make_cmd(0x05)
        cmd[4] = steps & 0xff
        cmd[5] = (steps >> 8)&0xff
        cmd[6] = self.shutter_open | (self.shutter_power<<1)
        cmd[8] = self.fanspeed
        cmd[10] = endzone & 0xff
        cmd[11] = (endzone>>8) & 0xff
        cmd[12] = backoff & 0xff
        cmd[13] = (backoff>>8) & 0xff
        cmd[14] = step_ontime & 0xff
        cmd[15] = (step_ontime>>8) & 0xff
        cmd[16] = step_offtime & 0xff
        cmd[17] = (step_offtime>>8) & 0xff
        self.send(cmd)

    def shutter(self, closed):
        cmd = self.make_cmd(0x05)
        self.shutter_open = 0 if closed else 1
        self.shutter_power = 1
        cmd[6] = self.shutter_open | (self.shutter_power<<1)
        cmd[8] = self.fanspeed
        self.send(cmd)
        time.sleep(1)
        self.shutter_open = 0
        self.shutter_power = 0
        cmd[6] = self.shutter_open | (self.shutter_power<<1)
        cmd[8] = self.fanspeed
        self.send(cmd)

    def fan(self, fanspeed):     # fanspeed is 0..15
        #print('speed: %d'%fanspeed)
        self.fanspeed = fanspeed
        cmd = self.make_cmd(0x85)
        cmd[6] = self.shutter_open | (self.shutter_power<<1)
        cmd[8] = self.fanspeed
        self.send(cmd)
        time.sleep(1)
        self.flush_rx_buf()

    def shutter_new(self, closed):
        cmd = self.make_cmd(0x08)
        cmd[1] = 0x01 if closed else 0x0
        self.send(cmd)

    def lf(self, val):
        cmd = self.make_cmd(0x09)
        cmd[1] = 0x01 if val else 0x0
        self.send(cmd)

    # read from housekeeping socket, wait for one from this quabo
    # (discard ones from other quabos)
    # wait for up to 10 sec
    # returns the HK packet, or None
    #
    def read_hk_packet(self):
        x = None
        end_time = time.time() + 10
        if not self.have_hk_sock:
            self.hk_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.hk_sock.settimeout(0.5)
            self.hk_sock.bind(("", UDP_HK_PORT))
            self.have_hk_sock = True

        while True:
            try:
                x = self.hk_sock.recvfrom(2048)
                # returns (data, (ip_addr, port))
            except:
                continue
            src = x[1]
            if src[0] == self.ip_addr:
                self.hk_sock.close()
                self.have_hk_sock = False
                return x[0]
            if time.time() > end_time:
                self.hk_sock.close()
                self.have_hk_sock = False
                return None

    # set destination IP addr for both PH and image packets
    #
    def data_packet_destination(self, ip_addr_str):
        ip_addr_bytes = util.ip_addr_str_to_bytes(ip_addr_str)
        cmd = self.make_cmd(0x0a)
        for i in range(4):
            cmd[i+1] = ip_addr_bytes[i]
            cmd[i+5] = ip_addr_bytes[i]
        self.flush_rx_buf()
        self.send(cmd)
        reply = self.sock.recvfrom(12)
        bytes = reply[0]
        count = len(bytes)
        #print('got %d bytes in reply'%count)
        if count != 12:
            return
        #print('Mac addr for PH packets: %s'%(util.mac_addr_str(bytes[0:6])))
        #print('Mac addr for image packets: %s'%(util.mac_addr_str(bytes[6:12])))

    def hk_packet_destination(self, ip_addr_str):
        ip_addr_bytes = util.ip_addr_str_to_bytes(ip_addr_str)
        cmd = self.make_cmd(0x0b)
        for i in range(4):
            cmd[i+1] = ip_addr_bytes[i]
        self.send(cmd)

# IMPLEMENTATION STUFF FOLLOWS

    def send(self, cmd):
        #print('sending %d bytes'%(len(cmd)))
        self.sock.sendto(bytes(cmd), (self.ip_addr, UDP_CMD_PORT))

    def make_cmd(self, cmd):
        x = bytearray(64)
        for i in range(64):
            x[i] = 0
        x[0] = cmd
        return x

    def flush_rx_buf(self):
        count = 0
        nbytes = 0
        while (count<32):
            try:
                x = self.sock.recvfrom(2048)
                # returns (data, ip_addr)
                nbytes += len(x[0])
                count += 1
            except:
                break
        #print('flush_rx_buffer: got %d bytes'%nbytes)

    def parse_hv_params(self, fhand, cmd):
        for line in fhand:
            if line.startswith("*"): continue
            #strip off the comment
            strippedline = line.split('*')[0]
            #Split the tag field from the cs value field
            fields = strippedline.split("=")
            if len(fields) !=2: continue
            tag = fields[0].strip()
            if (tag.startswith("HV")):
                chan = tag.split('_')[1]
                chan = int(chan)
                val = int(fields[1],0)
                self.HV_vals[chan]=val
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[2*chan+2]=LSbyte
                cmd[2*chan+3]=MSbyte

    def parse_trigger_mask(self, fhand, cmd):
        for line in fhand:
            if line.startswith("*"): continue
            #strip off the comment
            strippedline = line.split('*')[0]
            #Split the tag field from the cs value field
            fields = strippedline.split("=")
            if len(fields) !=2: continue
            tag = fields[0].strip()
            chan_mask = [0,0,0,0,0,0,0,0,0]
            if (tag.startswith("CHANMASK")):
                chan = tag.split('_')[1]
                chan = int(chan)
                val = int(fields[1],0)
                chan_mask[chan]=val
                for i in range (4):
                    cmd[4*chan+4]=val & 0xff
                    cmd[4*chan+5]=(val>>8) & 0xff
                    cmd[4*chan+6]=(val>>16) & 0xff
                    cmd[4*chan+7]=(val>>24) & 0xff

    def parse_acq_parameters(self, fhand, cmd):
        for line in fhand:
            if line.startswith("*"): continue
            #strip off the comment
            strippedline = line.split('*')[0]
            #Split the tag field from the cs value field
            fields = strippedline.split("=")
            if len(fields) !=2: continue
            tag = fields[0].strip()
            if (tag == "ACQMODE"):
                val = int(fields[1],0)
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[2]=LSbyte
                cmd[3]=MSbyte
            if (tag == "ACQINT"):
                val = int(fields[1],0)
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[4]=LSbyte
                cmd[5]=MSbyte
            if (tag == "HOLD1"):
                val = int(fields[1],0)
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[6]=LSbyte
                cmd[7]=MSbyte
            if (tag == "HOLD2"):
                val = int(fields[1],0)
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[8]=LSbyte
                cmd[9]=MSbyte
            if (tag == "ADCCLKPH"):
                val = int(fields[1],0)
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[10]=LSbyte
                cmd[11]=MSbyte
            if (tag == "MONCHAN"):
                val = int(fields[1],0)
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[12]=LSbyte
                cmd[13]=MSbyte
            if (tag == "STIMON"):
                val = int(fields[1],0)
                LSbyte = val & 0x01
                MSbyte = 0
                cmd[14]=LSbyte
                cmd[15]=MSbyte
            if (tag == "STIM_LEVEL"):
                val = int(fields[1],0)
                LSbyte = val & 0xff
                MSbyte = 0
                cmd[16]=LSbyte
                cmd[17]=MSbyte
            if (tag == "STIM_RATE"):
                val = int(fields[1],0)
                LSbyte = val & 0xff
                MSbyte = 0
                cmd[18]=LSbyte
                cmd[19]=MSbyte
            if (tag == "EN_WR_UART"):
                val = int(fields[1],0)
                LSbyte = val & 0x01
                MSbyte = 0
                cmd[20]=LSbyte
                cmd[21]=MSbyte
            if (tag == "FLASH_RATE"):
                val = int(fields[1],0)
                LSbyte = val & 0x07
                MSbyte = 0
                cmd[22]=LSbyte
                cmd[23]=MSbyte
            if (tag == "FLASH_LEVEL"):
                val = int(fields[1],0)
                LSbyte = val & 0x1f
                MSbyte = 0
                cmd[24]=LSbyte
                cmd[25]=MSbyte
            if (tag == "FLASH_WIDTH"):
                val = int(fields[1],0)
                LSbyte = val & 0x0f
                MSbyte = 0
                cmd[26]=LSbyte
                cmd[27]=MSbyte

    # given a config dictionary, make a MAROC config command
    #
    def make_maroc_cmd(self, config, cmd):
        cmd[0] = 0x01
        for tag, val in config.items():
            # Make a list of the should-be 4 ascii values
            vals = val.split(",")

            # Make a list of integers
            vals_int = []
            for i in range(len(vals)): vals_int.append(int(vals[i],0))

            # For each tag, set the appropriate bit field
            if (tag == "OTABG_ON"): self.set_bits_4(tag, vals_int, 0, 1)
            if (tag == "DAC_ON"): self.set_bits_4(tag, vals_int, 1, 1)
            if (tag == "SMALL_DAC"): self.set_bits_4(tag, vals_int, 2, 1)
            if (tag == "DAC2"):
                #need to reverse the bits
                vals_revbits = []
                for i in range (4):
                    vals_revbits.append(reverse_bits(int(vals[i],0),10))
                self.set_bits_4(tag, vals_revbits, 3, 10)
            if (tag == "DAC1"):
                vals_revbits = []
                for i in range (4):
                    vals_revbits.append(reverse_bits(int(vals[i],0),10))
                self.set_bits_4(tag, vals_revbits, 13, 10)
            if (tag == "ENB_OUT_ADC"): self.set_bits_4(tag, vals_int, 23, 1)
            if (tag == "INV_START_GRAY"): self.set_bits_4(tag, vals_int, 24, 1)
            if (tag == "RAMP8B"): self.set_bits_4(tag, vals_int, 25, 1)
            if (tag == "RAMP10B"): self.set_bits_4(tag, vals_int, 26, 1)
            if (tag == "CMD_CK_MUX"): self.set_bits_4(tag, vals_int, 155, 1)
            if (tag == "D1_D2"): self.set_bits_4(tag, vals_int, 156, 1)
            if (tag == "INV_DISCR_ADC"): self.set_bits_4(tag, vals_int, 157, 1)
            if (tag == "POLAR_DISCRI"): self.set_bits_4(tag, vals_int, 158, 1)
            if (tag == "ENB3ST"): self.set_bits_4(tag, vals_int, 159, 1)
            if (tag == "VAL_DC_FSB2"): self.set_bits_4(tag, vals_int, 160, 1)
            if (tag == "SW_FSB2_50F"): self.set_bits_4(tag, vals_int, 161, 1)
            if (tag == "SW_FSB2_100F"): self.set_bits_4(tag, vals_int, 162, 1)
            if (tag == "SW_FSB2_100K"): self.set_bits_4(tag, vals_int, 163, 1)
            if (tag == "SW_FSB2_50K"): self.set_bits_4(tag, vals_int, 164, 1)
            if (tag == "VALID_DC_FS"): self.set_bits_4(tag, vals_int, 165, 1)
            if (tag == "CMD_FSB_FSU"): self.set_bits_4(tag, vals_int, 166, 1)
            if (tag == "SW_FSB1_50F"): self.set_bits_4(tag, vals_int, 167, 1)
            if (tag == "SW_FSB1_100F"): self.set_bits_4(tag, vals_int, 168, 1)
            if (tag == "SW_FSB1_100K"): self.set_bits_4(tag, vals_int, 169, 1)
            if (tag == "SW_FSB1_50k"): self.set_bits_4(tag, vals_int, 170, 1)
            if (tag == "SW_FSU_100K"): self.set_bits_4(tag, vals_int, 171, 1)
            if (tag == "SW_FSU_50K"): self.set_bits_4(tag, vals_int, 172, 1)
            if (tag == "SW_FSU_25K"): self.set_bits_4(tag, vals_int, 173, 1)
            if (tag == "SW_FSU_40F"): self.set_bits_4(tag, vals_int, 174, 1)
            if (tag == "SW_FSU_20F"): self.set_bits_4(tag, vals_int, 175, 1)
            if (tag == "H1H2_CHOICE"): self.set_bits_4(tag, vals_int, 176, 1)
            if (tag == "EN_ADC"): self.set_bits_4(tag, vals_int, 177, 1)
            if (tag == "SW_SS_1200F"): self.set_bits_4(tag, vals_int, 178, 1)
            if (tag == "SW_SS_600F"): self.set_bits_4(tag, vals_int, 179, 1)
            if (tag == "SW_SS_300F"): self.set_bits_4(tag, vals_int, 180, 1)
            if (tag == "ON_OFF_SS"): self.set_bits_4(tag, vals_int, 181, 1)
            if (tag == "SWB_BUF_2P"): self.set_bits_4(tag, vals_int, 182, 1)
            if (tag == "SWB_BUF_1P"): self.set_bits_4(tag, vals_int, 183, 1)
            if (tag == "SWB_BUF_500F"): self.set_bits_4(tag, vals_int, 184, 1)
            if (tag == "SWB_BUF_250F"): self.set_bits_4(tag, vals_int, 185, 1)
            if (tag == "CMD_FSB"): self.set_bits_4(tag, vals_int, 186, 1)
            if (tag == "CMD_SS"): self.set_bits_4(tag, vals_int, 187, 1)
            if (tag == "CMD_FSU"): self.set_bits_4(tag, vals_int, 188, 1)

            #Look for a MASKOR1 value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("MASKOR1"):
                chan = tag.split('_')[1]
                chan = int(chan)
                self.set_bits_4(tag, vals_int, 154-(2*chan), 1)
            #Look for a MASKOR2 value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("MASKOR2"):
                chan = tag.split('_')[1]
                chan = int(chan)
                self.set_bits_4(tag, vals_int, 153-(2*chan), 1)
            #Look for a CTEST value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("CTEST"):
                chan = tag.split('_')[1]
                chan = int(chan)
                #if chan in range(4):
                    #vals_int = [0,0,0,0]
                self.set_bits_4(tag, vals_int, 828-chan, 1)
                #print(tag, vals_int, chan)

            #Look for a GAIN value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("GAIN"):
                chan = tag.split('N')[1]
                chan = int(chan)
                #Another list, with integer values, bits reversed
                vals_revbits = []
                for i in range (4):
                    vals_revbits.append(reverse_bits((vals_int[i]),8))
                self.set_bits_4(tag, vals_revbits, 757-9*chan,8)
            for ii in range(104):
                cmd[ii+4] = self.MAROC_regs[0][ii]
                cmd[ii+132] = self.MAROC_regs[1][ii]
                cmd[ii+260] = self.MAROC_regs[2][ii]
                cmd[ii+388] = self.MAROC_regs[3][ii]

    # Set bits in MAROC_regs[chip] according to the input values.
    # Maximum value for field_width is 16 (a value can only span three bytes)
    #
    def set_bits(self, chip, lsb_pos, field_width, value):
        if (field_width >16): return
        if ((field_width + lsb_pos) > SERIAL_COMMAND_LENGTH): return
        shift = (lsb_pos % 8)
        byte_pos = int((lsb_pos+7-shift)/8)
        mask=0
        for ii in range(0, field_width):
            mask = mask << 1
            mask = (mask | 0x1)
        mask = mask << shift

        self.MAROC_regs[chip][byte_pos] = self.MAROC_regs[chip][byte_pos] & ((~mask) & 0xff)
        self.MAROC_regs[chip][byte_pos] = self.MAROC_regs[chip][byte_pos] | ((value << shift) & 0xff)
        #if field spans a byte boundary
        if ((shift + field_width) > 8):
            self.MAROC_regs[chip][byte_pos + 1] = self.MAROC_regs[chip][byte_pos + 1] & ((~(mask>>8)) & 0xff)
            self.MAROC_regs[chip][byte_pos + 1] = self.MAROC_regs[chip][byte_pos + 1] | (((value >> (8-shift))) & 0xff)
        if ((shift + field_width) > 16):
            self.MAROC_regs[chip][byte_pos + 2] = self.MAROC_regs[chip][byte_pos + 2] & ((~(mask>>16)) & 0xff)
            self.MAROC_regs[chip][byte_pos + 2] = self.MAROC_regs[chip][byte_pos + 2] | (((value >> (16-shift))) & 0xff)

    # take a 4-element list and call set_bits for each MAROC
    #
    def set_bits_4(self, tag, vals, lsb_pos, field_width):
        #vals = instring.split(",")
        if (len(vals) != 4):
            raise Exception("need 4 elements for " + tag +"\n")
        self.set_bits(0, lsb_pos, field_width, vals[0])
        self.set_bits(1, lsb_pos, field_width, vals[1])
        self.set_bits(2, lsb_pos, field_width, vals[2])
        self.set_bits(3, lsb_pos, field_width, vals[3])

# END OF CLASS QUABO

# read a file of the form
# name0=val0
# name1=val1
# ... and return a dictionary mapping name to value.
# strip off comments (text starting with *)
#
def parse_quabo_config_file(path):
    x = {}
    with open(path) as f:
        for line in f:
            if line.startswith("*"):
                continue

            # strip off the comment
            strippedline = line.split('*')[0]
            
            # Split the tag field from the cs value field
            fields = strippedline.split("=")
            if len(fields) !=2:
                continue
            name = fields[0].strip()
            val = fields[1].strip()
            x[name] = val
    return x

def reverse_bits(data_in, width):
    data_out = 0
    for ii in range(width):
        data_out = data_out << 1
        if (data_in & 1): data_out = data_out | 1
        data_in = data_in >> 1
    return data_out

# write maroc config cmd to a file
def write_maroc_config_cmd():
    q = QUABO('1.1.1.1')
    config = parse_quabo_config_file('quabo_config.txt')
    cmd = bytearray(492)
    q.make_maroc_cmd(config, cmd)
    with open('maroc_cmd_new.bin', 'w') as f:
        f.write(cmd)

if __name__ == "__main__":
# test stuff goes here

    write_maroc_config_cmd();

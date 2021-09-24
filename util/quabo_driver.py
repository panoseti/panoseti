# class for communicating with a quabo board
#
# example:
#   q = QUABO("128.5.4.2")
#   q.lf(False)
#
# A QUABO object has info about the quabo state,
# e.g. MAROC regs, shutter status etc.
# Currently these are initialized to zero.
# Ideally we should get the actual values from the quabo.
#
# See https://github.com/panoseti/panoseti/wiki/Quabo-device-driver

import socket, time

UDP_CMD_PORT= 60000
    # port used on both sides

SERIAL_COMMAND_LENGTH = 829

class QUABO:
    def __init__(self, ip_addr):
        self.ip_addr = ip_addr;
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        self.sock.bind(("", UDP_CMD_PORT))
        self.shutter_open = 0
        self.shutter_power = 0
        self.fanspeed = 0
        self.HV_vals = [0,0,0,0]
        self.MAROC_regs = []
        for i in range (4):
            self.MAROC_regs.append([0 for x in range(104)])

    def close(self):
        self.sock.close();

    def send_maroc_params(self, fhand):            # fhand is open config file
        cmd = bytearray(492)
        cmd[0] = 0x01
        self.parse_maroc_params(fhand, cmd)
        self.flush_rx_buf()
        self.send(cmd);

    def calibrate_ph_baseline(self, fhand):        # fhand is output file handle
        cmd = self.make_cmd(0x07)
        self.flush_rx_buf()
        self.send(cmd)
        time.sleep(2)
        reply = self.sock.recvfrom(1024)
        bytesback = reply[0]
        now =time.ctime().split(" ")[3]
        fhand.write(str(now) + ',')
        for n in range(256):
            val=bytesback[2*n+4]+256*bytesback[2*n+5]
            fhand.write(str(val) + ',')
        fhand.write('\n')

    def hv_config(self, fhand):
        cmd = self.make_cmd(0x02)
        self.parse_hv_params(fhand, cmd)
        self.flush_rx_buf()     # needed?
        self.send(cmd);

    def hv_chan(self, chan, value):
        cmd = self.make_cmd(0x02)
        self.HV_vals[chan] = value
        for i in range(4):
            LSbyte = HV_vals[i] & 0xff
            MSbyte = (HV_vals[i] >> 8) & 0xff
            cmd[2*i+2]=LSbyte
            cmd[2*i+3]=MSbyte
        self.send(cmd);

    def hv_zero(self):
        cmd = self.make_cmd(0x02)
        for i in range(4):
            cmd[2*i+2]=0
            cmd[2*i+3]=0
        self.send(cmd);

    def send_acq_parameters(self, fhand):        # fhand is open config file
        cmd = self.make_cmd(0x03)
        self.parse_acq_parameters(fhand, cmd)
        self.flush_rx_buf()
        self.send(cmd);

    def send_trigger_mask(self, fhand):          # fhand is open config file
        cmd = self.make_cmd(0x06)
        self.parse_trigger_mask(fhand, cmd)
        self.flush_rx_buf()
        self.send(cmd);

    def reset(self):
        cmd = self.make_cmd(0x04)
        self.send(cmd);

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
        self.send(cmd);

    def shutter(self, closed):
        cmd = self.make_cmd(0x05)
        self.shutter_open = 0 if closed else 1
        self.shutter_power = 1
        cmd[6] = self.shutter_open | (self.shutter_power<<1)
        cmd[8] = self.fanspeed
        self.send(cmd);
        time.sleep(1);
        self.shutter_open = 0
        self.shutter_power = 0
        cmd[6] = self.shutter_open | (self.shutter_power<<1)
        cmd[8] = self.fanspeed
        self.send(cmd);

    def fan(self, fanspeed):     # fanspeed is 0..15
        self.fanspeed = fanspeed
        cmd = self.make_cmd(0x05)
        cmd[6] = self.shutter_open | (self.shutter_power<<1)
        cmd[8] = self.fanspeed
        self.send(cmd)

    def shutter_new(self, closed):
        cmd = self.make_cmd(0x08)
        cmd[1] = 0x01 if closed else 0x0;
        self.send(cmd)

    def lf(self, val):
        cmd = self.make_cmd(0x09)
        cmd[1] = 0x01 if val else 0x0;
        self.send(cmd)

# IMPLEMENTATION STUFF FOLLOWS

    def send(self, cmd):
        self.sock.sendto(bytes(cmd), (self.ip_addr, UDP_CMD_PORT))

    def make_cmd(self, cmd):
        x = bytearray(64)
        for i in range(64):
            x[i] = 0
        x[0] = cmd
        return x

    def flush_rx_buf(self):
        count = 0
        while (count<32):
            try:
                dumpbytes = self.sock.recvfrom(2048)
                count += 1
            except:
                break

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

    def parse_maroc_params(self, fhand, cmd):
        for line in fhand:
            if line.startswith("*"):
                continue

            # strip off the comment
            strippedline = line.split('*')[0]
            
            # Split the tag field from the cs value field
            fields = strippedline.split("=")
            if len(fields) !=2:
                continue
            tag = fields[0].strip()

            # Make a list of the should-be 4 ascii values
            vals = fields[1].split(",")

            # Make a list of integers
            vals_int = []
            for i in range(len(vals)): vals_int.append(int(vals[i],0))

            # For each tag, set the appropriate bit field
            if (tag == "OTABG_ON"): self.set_bits_4(fields[0], vals_int, 0, 1)
            if (tag == "DAC_ON"): self.set_bits_4(fields[0], vals_int, 1, 1)
            if (tag == "SMALL_DAC"): self.set_bits_4(fields[0], vals_int, 2, 1)
            if (tag == "DAC2"):
                #need to reverse the bits
                vals_revbits = []
                for i in range (4):
                    vals_revbits.append(reverse_bits(int(vals[i],0),10))
                self.set_bits_4(fields[0], vals_revbits, 3, 10)
            if (tag == "DAC1"):
                vals_revbits = []
                for i in range (4):
                    vals_revbits.append(reverse_bits(int(vals[i],0),10))
                self.set_bits_4(fields[0], vals_revbits, 13, 10)
            if (tag == "ENB_OUT_ADC"): self.set_bits_4(fields[0], vals_int, 23, 1)
            if (tag == "INV_START_GRAY"): self.set_bits_4(fields[0], vals_int, 24, 1)
            if (tag == "RAMP8B"): self.set_bits_4(fields[0], vals_int, 25, 1)
            if (tag == "RAMP10B"): self.set_bits_4(fields[0], vals_int, 26, 1)
            if (tag == "CMD_CK_MUX"): self.set_bits_4(fields[0], vals_int, 155, 1)
            if (tag == "D1_D2"): self.set_bits_4(fields[0], vals_int, 156, 1)
            if (tag == "INV_DISCR_ADC"): self.set_bits_4(fields[0], vals_int, 157, 1)
            if (tag == "POLAR_DISCRI"): self.set_bits_4(fields[0], vals_int, 158, 1)
            if (tag == "ENB3ST"): self.set_bits_4(fields[0], vals_int, 159, 1)
            if (tag == "VAL_DC_FSB2"): self.set_bits_4(fields[0], vals_int, 160, 1)
            if (tag == "SW_FSB2_50F"): self.set_bits_4(fields[0], vals_int, 161, 1)
            if (tag == "SW_FSB2_100F"): self.set_bits_4(fields[0], vals_int, 162, 1)
            if (tag == "SW_FSB2_100K"): self.set_bits_4(fields[0], vals_int, 163, 1)
            if (tag == "SW_FSB2_50K"): self.set_bits_4(fields[0], vals_int, 164, 1)
            if (tag == "VALID_DC_FS"): self.set_bits_4(fields[0], vals_int, 165, 1)
            if (tag == "CMD_FSB_FSU"): self.set_bits_4(fields[0], vals_int, 166, 1)
            if (tag == "SW_FSB1_50F"): self.set_bits_4(fields[0], vals_int, 167, 1)
            if (tag == "SW_FSB1_100F"): self.set_bits_4(fields[0], vals_int, 168, 1)
            if (tag == "SW_FSB1_100K"): self.set_bits_4(fields[0], vals_int, 169, 1)
            if (tag == "SW_FSB1_50k"): self.set_bits_4(fields[0], vals_int, 170, 1)
            if (tag == "SW_FSU_100K"): self.set_bits_4(fields[0], vals_int, 171, 1)
            if (tag == "SW_FSU_50K"): self.set_bits_4(fields[0], vals_int, 172, 1)
            if (tag == "SW_FSU_25K"): self.set_bits_4(fields[0], vals_int, 173, 1)
            if (tag == "SW_FSU_40F"): self.set_bits_4(fields[0], vals_int, 174, 1)
            if (tag == "SW_FSU_20F"): self.set_bits_4(fields[0], vals_int, 175, 1)
            if (tag == "H1H2_CHOICE"): self.set_bits_4(fields[0], vals_int, 176, 1)
            if (tag == "EN_ADC"): self.set_bits_4(fields[0], vals_int, 177, 1)
            if (tag == "SW_SS_1200F"): self.set_bits_4(fields[0], vals_int, 178, 1)
            if (tag == "SW_SS_600F"): self.set_bits_4(fields[0], vals_int, 179, 1)
            if (tag == "SW_SS_300F"): self.set_bits_4(fields[0], vals_int, 180, 1)
            if (tag == "ON_OFF_SS"): self.set_bits_4(fields[0], vals_int, 181, 1)
            if (tag == "SWB_BUF_2P"): self.set_bits_4(fields[0], vals_int, 182, 1)
            if (tag == "SWB_BUF_1P"): self.set_bits_4(fields[0], vals_int, 183, 1)
            if (tag == "SWB_BUF_500F"): self.set_bits_4(fields[0], vals_int, 184, 1)
            if (tag == "SWB_BUF_250F"): self.set_bits_4(fields[0], vals_int, 185, 1)
            if (tag == "CMD_FSB"): self.set_bits_4(fields[0], vals_int, 186, 1)
            if (tag == "CMD_SS"): self.set_bits_4(fields[0], vals_int, 187, 1)
            if (tag == "CMD_FSU"): self.set_bits_4(fields[0], vals_int, 188, 1)

            #Look for a MASKOR1 value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("MASKOR1"):
                chan = tag.split('_')[1]
                chan = int(chan)
                self.set_bits_4(fields[0], vals_int, 154-(2*chan), 1)
            #Look for a MASKOR2 value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("MASKOR2"):
                chan = tag.split('_')[1]
                chan = int(chan)
                self.set_bits_4(fields[0], vals_int, 153-(2*chan), 1)
            #Look for a CTEST value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("CTEST"):
                chan = tag.split('_')[1]
                chan = int(chan)
                #if chan in range(4):
                    #vals_int = [0,0,0,0]
                self.set_bits_4(fields[0], vals_int, 828-chan, 1)
                #print(fields[0], vals_int, chan)

            #Look for a GAIN value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("GAIN"):
                chan = tag.split('N')[1]
                chan = int(chan)
                #Another list, with integer values, bits reversed
                vals_revbits = []
                for i in range (4):
                    vals_revbits.append(reverse_bits((vals_int[i]),8))
                self.set_bits_4(fields[0], vals_revbits, 757-9*chan,8)
            for ii in range(104):
                cmd[ii+4] = self.MAROC_regs[0][ii]
                cmd[ii+132] = self.MAROC_regs[1][ii]
                cmd[ii+260] = self.MAROC_regs[2][ii]
                cmd[ii+388] = self.MAROC_regs[3][ii]

    # Set bits in the command_buf according to the input values.
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

    #take a 4-element list and call set_bits 4 times
    def set_bits_4(self, tag, vals, lsb_pos, field_width):
        #vals = instring.split(",")
        if (len(vals) != 4):
            print("need 4 elements for " + tag +"\n")
            return
        self.set_bits(0, lsb_pos, field_width, vals[0])
        self.set_bits(1, lsb_pos, field_width, vals[1])
        self.set_bits(2, lsb_pos, field_width, vals[2])
        self.set_bits(3, lsb_pos, field_width, vals[3])

def reverse_bits(data_in, width):
    data_out = 0
    for ii in range(width):
        data_out = data_out << 1
        if (data_in & 1): data_out = data_out | 1
        data_in = data_in >> 1
    return data_out

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

from __future__ import annotations

import json
import socket
import time
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import Any

from panoseti_grpc.telemetry.logger import get_logger
from pydantic import IPvAnyAddress

from control.utils import util
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import DataConfig

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

QUABO_CONFIG_FILE = 'quabo_config.txt'
QUABO_CONFIG_FILE_TEMPLATE = 'quabo_config_{ip_addr}.json'


class DAQ_PARAMS:
    def __init__(self, do_image: bool, image_us: int, image_8bit: bool, do_ph: bool, bl_subtract: bool, do_any_trigger: bool = False, do_group_ph_frames: bool = False) -> None:
        self.do_image = do_image
        self.image_us = image_us
        self.image_8bit = image_8bit
        self.do_ph = do_ph
        self.bl_subtract = bl_subtract
        self.do_any_trigger = do_any_trigger
        self.do_group_ph_frames = do_group_ph_frames
        self.do_flash = False
        self.do_stim = False
        self.flash_rate: int = 0
        self.flash_level: int = 0
        self.flash_width: int = 0
        self.stim_rate: int = 0
        self.stim_level: int = 0

    def set_flash_params(self, rate: int, level: int, width: int) -> None:
        self.do_flash = True
        self.flash_rate = rate
        self.flash_level = level
        self.flash_width = width
    
    def set_stim_params(self, rate: int, level: int) -> None:
        self.do_stim = True
        self.stim_rate = rate
        self.stim_level = level

class QUABO:
    def __init__(self, ip_addr: IPvAnyAddress | str, port: int = UDP_CMD_PORT, config_file_path: str = 'quabo_config.txt') -> None:
        if isinstance(ip_addr, (str, IPv4Address, IPv6Address)):
            validated_ip_addr = ip_address(ip_addr)
        else:
            raise ValueError(f"Unexpected type for ip_addr: {type(ip_addr)=}, which should be IPvAnyAddress or str")
        self.ip_addr = str(validated_ip_addr)
        self.port = port
        self.config_file_path = config_file_path

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        self.have_hk_sock = False
        self.hk_sock: socket.socket | None = None

        self.shutter_open = 0
        self.shutter_power = 0
        self.fanspeed = 0
        self.HV_vals: list[int] = [0, 0, 0, 0]
        self.MAROC_regs: list[list[int]] = []
        for _i in range(4):
            self.MAROC_regs.append([0 for _x in range(104)])
        # create a logger
        PanoPaths.logs_dir().mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(service_name='quabo_driver', log_dir=str(PanoPaths.logs_dir()), grpc_enabled=True)
        # self.logger.info('************************************')

    def close(self) -> None:
        self.logger.info('close')
        self.sock.close()

    def send_daq_params(self, params: DAQ_PARAMS, echo: bool = False) -> None:
        self.logger.info('send_daq_params')
        cmd = self.make_cmd(0x83 if echo else 0x03)
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
        cmd[12] = 69
        if params.do_flash:
            self.logger.info('flash is on.')
            self.logger.debug(f'flash_rate: {params.flash_rate}')
            self.logger.debug(f'flash_level: {params.flash_level}')
            self.logger.debug(f'flash_width: {params.flash_width}')
            cmd[22] = params.flash_rate
            cmd[24] = params.flash_level
            cmd[26] = params.flash_width
        else:
            self.logger.info('flash is off.')
        if params.do_stim:
            self.logger.info('STIM is on.')
            self.logger.debug(f'stim_level: {params.stim_level}')
            self.logger.debug(f'stim_rate: {params.stim_rate}')
            cmd[14] = 1
            cmd[16] = params.stim_level
            cmd[18] = params.stim_rate
        else:
            self.logger.info('STIM is off.')
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def send_maroc_params_file(self) -> None:
        self.logger.info('send_maroc_params_file')
        cmd = bytearray(492)
        with open(self.config_file_path):
            config = parse_quabo_config_file(self.config_file_path)
        self.make_maroc_cmd(config, cmd)
        self.flush_rx_buf()
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def send_maroc_params(self, config: dict[str, Any]) -> None:
        self.logger.info('send_maroc_params')
        cmd = bytearray(492)
        self.make_maroc_cmd(config, cmd)
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    # returns the list of 256 coefficients
    #
    def calibrate_ph_baseline(self) -> list[int]:
        self.logger.info('calibrate_ph_baseline')
        # make the quabos send out some ph packets
        daq_start = DAQ_PARAMS(
                do_image=False,
                image_us=4999,
                image_8bit=False,
                do_ph=True,
                bl_subtract=False
            )
        daq_stop = DAQ_PARAMS(False, 0, False, False, True)
        # Use local IP as destination for the calibration packets
        try:
            from control.utils.util import local_ip
            ip_addr = ip_address(local_ip()[0])
        except Exception:
            ip_addr = ip_address('0.0.0.0')
        self.data_packet_destination(ip_addr)
        self.send_daq_params(daq_start)
        #time.sleep(1)
        self.send_daq_params(daq_stop)
        cmd = self.make_cmd(0x07)
        self.flush_rx_buf()
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)
        time.sleep(2)
        reply = self.sock.recvfrom(1024)
        bytesback = reply[0]
        x: list[int] = []
        for n in range(256):
            val = bytesback[2*n+4] + 256*bytesback[2*n+5]
            x.append(val)
        return x

    def hv_config(self, echo: bool = False) -> None:
        self.logger.info('hv_config')
        cmd = self.make_cmd(0x82 if echo else 0x02)
        with open(self.config_file_path) as f:
            self.parse_hv_params(f, cmd)
        self.flush_rx_buf()     # needed?
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def hv_set_chan(self, chan: int, value: int, echo: bool = False) -> None:
        self.logger.info(f'hv_set_chan: ch - {chan}, val - {value}')
        cmd = self.make_cmd(0x82 if echo else 0x02)
        self.HV_vals[chan] = int(value)
        for i in range(4):
            LSbyte = self.HV_vals[i] & 0xff
            MSbyte = (self.HV_vals[i] >> 8) & 0xff
            cmd[2*i+2]=LSbyte
            cmd[2*i+3]=MSbyte
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    # set high voltage for all 4 channels
    #
    def hv_set(self, values: list[int], echo: bool = False) -> None:
        self.logger.info(f'hv_set: val - {values[0]} {values[1]} {values[2]} {values[3]}')
        cmd = self.make_cmd(0x82 if echo else 0x02)
        for i in range(4):
            cmd[2*i+2] = values[i] & 0xff
            cmd[2*i+3] = (values[i] >> 8) & 0xff
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def send_acq_parameters_file(self, echo: bool = False) -> None:
        self.logger.info('send_acq_parameters_file')
        cmd = self.make_cmd(0x83 if echo else 0x03)
        with open(self.config_file_path) as f:
            self.parse_acq_parameters(f, cmd)
        self.flush_rx_buf()
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def send_trigger_mask(self, config: dict[str, int], do_flush_rx_buf: bool = True) -> None:
        self.logger.info('send_trigger_mask')
        cmd = self.make_cmd(0x06)
        self.make_trigger_mask_cmd(config, cmd)
        if do_flush_rx_buf:
            self.flush_rx_buf()
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def send_goe_mask(self, config: dict[str, int], do_flush_rx_buf: bool = True) -> None:
        self.logger.info('send_goe_mask')
        cmd = self.make_cmd(0x0e)
        self.make_goe_mask_cmd(config, cmd)
        if do_flush_rx_buf:
            self.flush_rx_buf()
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)
        
    def reset(self) -> None:
        self.logger.info('reset')
        cmd = self.make_cmd(0x04)
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def focus(self, steps: int) -> None:      # 1 to 50000, 0 to recalibrate
        self.logger.info(f'focus: steps - {steps}')
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
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def shutter(self, closed: bool) -> None:
        self.logger.info(f'shutter: {closed}')
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
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def fan(self, fanspeed: int) -> None:     # fanspeed is 0..15
        self.logger.info(f'fan: speed - {fanspeed}')
        self.fanspeed = fanspeed
        cmd = self.make_cmd(0x85)
        cmd[6] = self.shutter_open | (self.shutter_power<<1)
        cmd[8] = self.fanspeed
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)
        time.sleep(1)
        self.flush_rx_buf()

    def shutter_new(self, closed: bool) -> None:
        self.logger.info(f'shutter_new: {closed}')
        cmd = self.make_cmd(0x08)
        cmd[1] = 0x01 if closed else 0x0
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def lf(self, val: bool) -> None:
        self.logger.info(f'lf: val - {val}')
        cmd = self.make_cmd(0x09)
        cmd[1] = 0x01 if val else 0x0
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def swpps(self) -> None:
        self.logger.info('set software 1PPS.')
        cmd = self.make_cmd(0x0f)
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

    def write_maroc_config(self, config: dict[str, Any], config_path: Path) -> dict[str, Any]:
        self.logger.info(f'write_maroc_config to/from {config_path=}')
        try:
            with open(config_path, 'rb') as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            cfg = {}
        # create the tag list
        tag_list = ['OTABG_ON'      , 'DAC_ON'      , 'SMALL_DAC'       , 'DAC2'    ,
                    'DAC1'          , 'ENB_OUT_ADC' , 'INV_START_GRAY'  , 'RAMP8B'  ,
                    'RAMP10B'       , 'CMD_CK_MUX'  , 'D1_D2'           , 'INV_DISCR_ADC',
                    'POLAR_DISCRI'  , 'ENB3ST'      , 'VAL_DC_FSB2'     , 'SW_FSB2_50F',
                    'SW_FSB2_100F'  , 'SW_FSB2_100K', 'SW_FSB2_50K'     , 'VALID_DC_FS',
                    'CMD_FSB_FSU'   , 'SW_FSB1_50F' , 'SW_FSB1_100F'    , 'SW_FSB1_100K',
                    'SW_FSB1_50k'   , 'SW_FSU_100K' , 'SW_FSU_50K'      , 'SW_FSU_25K',
                    'SW_FSU_40F'    , 'SW_FSU_20F'  , 'H1H2_CHOICE'     , 'EN_ADC',
                    'SW_SS_1200F'   , 'SW_SS_600F'  , 'SW_SS_300F'      , 'ON_OFF_SS',
                    'SWB_BUF_2P'    , 'SWB_BUF_1P'  , 'SWB_BUF_500F'    , 'SWB_BUF_250F',
                    'CMD_FSB'       , 'CMD_SS'      , 'CMD_FSU']            
        # add GAIN to the tag list
        for i in range(64):
            tag_list.append(f'GAIN{i}')
        # add CTEST to the tag list
        for i in range(64):
            tag_list.append(f'CTEST_{i}')
        # add MASKOR1 to the tag list
        for i in range(64):
            tag_list.append(f'MASKOR1_{i}')
        # added MASKOR2 to the tag list
        for i in range(64):
            tag_list.append(f'MASKOR2_{i}')
        
        # get the maroc config params from config
        for tag in tag_list:
            cfg[tag] = config[tag]
        # write the maroc config params back to config file
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        return cfg
    
    def write_trigger_mask_config(self, config: dict[str, int], config_path: Path) -> dict[str, int]:
        self.logger.info(f'write_trigger_mask_config to/from {config_path=}')
        try:
            with open(config_path, 'rb') as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            cfg = {}
        # create the tag list
        tag_list: list[str] = []
        for i in range(9):
            tag_list.append(f'CHANMASK_{i}')
        # get trigger mask params from config
        for tag in tag_list:
            cfg[tag] = hex(config[tag])
        # write the trigger mask params back to config file
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        return cfg

    def write_goe_mask_config(self, config: dict[str, int], config_path: Path) -> dict[str, int]:
        self.logger.info(f'write_goe_mask_config to/from {config_path=}')
        try:
            with open(config_path, 'rb') as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            cfg = {}
        # create the tag list
        tag = 'GOEMASK'
        cfg[tag] = hex(config[tag])
        # write the trigger mask params back to config file
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        return cfg

    # read from housekeeping socket, wait for one from this quabo
    # (discard ones from other quabos)
    # wait for up to 10 sec
    # returns the HK packet, or None
    #
    def read_hk_packet(self) -> bytes | None:
        self.logger.info('read_hk_packet')
        x: Any = None
        end_time = time.time() + 10
        if not self.have_hk_sock:
            self.hk_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.hk_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                self.hk_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self.hk_sock.settimeout(0.5)
            self.hk_sock.bind(("", UDP_HK_PORT))
            self.have_hk_sock = True

        while True:
            if time.time() > end_time:
                if self.hk_sock:
                    self.hk_sock.close()
                self.have_hk_sock = False
                return None
            try:
                if self.hk_sock is None:
                    continue
                x = self.hk_sock.recvfrom(2048)
                # returns (data, (ip_addr, port))
            except (TimeoutError, OSError):
                continue
            src = x[1]
            if src[0] == self.ip_addr:
                if self.hk_sock:
                    self.hk_sock.close()
                self.have_hk_sock = False
                return x[0]

    # set destination IP addr for both PH and image packets
    #
    def data_packet_destination(self, ip_addr: IPvAnyAddress) -> bool:
        ip_addr_str = str(ip_addr)
        self.logger.info(f'data_packet_destination: {ip_addr_str}')
        # get the IP address from hostname
        ip_addr_str = socket.gethostbyname(ip_addr_str)
        ip_addr_bytes = util.ip_addr_str_to_bytes(ip_address(ip_addr_str))
        cmd = self.make_cmd(0x0a)
        for i in range(4):
            cmd[i+1] = ip_addr_bytes[i]
            cmd[i+5] = ip_addr_bytes[i]
        self.flush_rx_buf()
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)
        try:
            reply = self.sock.recvfrom(12)
            bytes_back = reply[0]
            count = len(bytes_back)
        except (TimeoutError, OSError):
            count = 0
        return count == 12

    def hk_packet_destination(self, ip_addr: IPvAnyAddress) -> None:
        ip_addr_str = str(ip_addr)
        self.logger.info(f'hk_packet_destination: {ip_addr_str}')
        # get the IP address from hostname
        ip_addr_str = socket.gethostbyname(ip_addr_str)
        ip_addr_bytes = util.ip_addr_str_to_bytes(ip_address(ip_addr_str))
        cmd = self.make_cmd(0x0b)
        for i in range(4):
            cmd[i+1] = ip_addr_bytes[i]
        self.logger.debug("CMD (spaced): " + ' '.join(f'{b:02X}' for b in cmd))
        self.send(cmd)

# IMPLEMENTATION STUFF FOLLOWS

    def send(self, cmd: bytearray | list[int]) -> None:
        self.logger.debug(f'send: {len(cmd)} bytes')
        self.sock.sendto(bytes(cmd), (self.ip_addr, self.port))

    def make_cmd(self, cmd: int) -> bytearray:
        x = bytearray(64)
        for i in range(64):
            x[i] = 0
        x[0] = cmd
        return x

    def flush_rx_buf(self) -> None:
        count = 0
        nbytes = 0
        while (count<32):
            try:
                x = self.sock.recvfrom(2048)
                # returns (data, ip_addr)
                nbytes += len(x[0])
                count += 1
            except (TimeoutError, OSError):
                break
        #print('flush_rx_buffer: got %d bytes'%nbytes)

    def parse_hv_params(self, fhand: Any, cmd: bytearray) -> None:
        self.logger.info('parse_hv_params')
        for line in fhand:
            if line.startswith("*"):
                continue
            #strip off the comment
            strippedline = line.split('*')[0]
            #Split the tag field from the cs value field
            fields = strippedline.split("=")
            if len(fields) !=2:
                continue
            tag = fields[0].strip()
            if (tag.startswith("HV")):
                chan = tag.split('_')[1]
                chan_int = int(chan)
                val = int(fields[1],0)
                self.logger.debug(f'chan - {chan_int}, val - {val}')
                self.HV_vals[chan_int]=val
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[2*chan_int+2]=LSbyte
                cmd[2*chan_int+3]=MSbyte

    def parse_trigger_mask(self, fhand: Any, cmd: bytearray) -> None:
        self.logger.info('parse_trigger_mask')
        for line in fhand:
            if line.startswith("*"):
                continue
            #strip off the comment
            strippedline = line.split('*')[0]
            #Split the tag field from the cs value field
            fields = strippedline.split("=")
            if len(fields) !=2:
                continue
            tag = fields[0].strip()
            chan_mask = [0,0,0,0,0,0,0,0,0]
            if (tag.startswith("CHANMASK")):
                chan = tag.split('_')[1]
                chan_int = int(chan)
                val = int(fields[1],0)
                chan_mask[chan_int]=val
                self.logger.debug(f'chan - {chan_int}, val - 0x{val:x}')
                for _i in range (4):
                    cmd[4*chan_int+4]=val & 0xff
                    cmd[4*chan_int+5]=(val>>8) & 0xff
                    cmd[4*chan_int+6]=(val>>16) & 0xff
                    cmd[4*chan_int+7]=(val>>24) & 0xff

    def parse_goe_mask(self, fhand: Any, cmd: bytearray) -> None:
        self.logger.info('parse_goe_mask')
        for line in fhand:
            if line.startswith("*"):
                continue
            #strip off the comment
            strippedline = line.split('*')[0]
            #Split the tag field from the cs value field
            fields = strippedline.split("=")
            if len(fields) !=2:
                continue
            tag = fields[0].strip()
            if (tag.startswith("GOEMASK")):
                val = int(fields[1],0)
                cmd[4] = val & 0x03
                self.logger.debug('goe mask val - 0x%x'%(val & 0x03))

    def parse_acq_parameters(self, fhand: Any, cmd: bytearray) -> None:
        self.logger.info('parse_acq_paramters')
        for line in fhand:
            if line.startswith("*"):
                continue
            #strip off the comment
            strippedline = line.split('*')[0]
            #Split the tag field from the cs value field
            fields = strippedline.split("=")
            if len(fields) !=2:
                continue
            tag = fields[0].strip()
            if (tag == "ACQMODE"):
                val = int(fields[1],0)
                self.logger.debug(f'ACQMODE - 0x{val:x}')
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[2]=LSbyte
                cmd[3]=MSbyte
            if (tag == "ACQINT"):
                val = int(fields[1],0)
                self.logger.debug(f'ACQINT - {val}')
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[4]=LSbyte
                cmd[5]=MSbyte
            if (tag == "HOLD1"):
                val = int(fields[1],0)
                self.logger.debug(f'HOLD1 - {val}')
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[6]=LSbyte
                cmd[7]=MSbyte
            if (tag == "HOLD2"):
                val = int(fields[1],0)
                self.logger.debug(f'HOLD2 - {val}')
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[8]=LSbyte
                cmd[9]=MSbyte
            if (tag == "ADCCLKPH"):
                val = int(fields[1],0)
                self.logger.debug(f'ADCCLKPH - {val}')
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[10]=LSbyte
                cmd[11]=MSbyte
            if (tag == "MONCHAN"):
                val = int(fields[1],0)
                self.logger.debug(f'MONCHAN   - {val}')
                LSbyte = val & 0xff
                MSbyte = (val >> 8) & 0xff
                cmd[12]=LSbyte
                cmd[13]=MSbyte
            if (tag == "STIMON"):
                val = int(fields[1],0)
                self.logger.debug(f'STIMON - {val}')
                LSbyte = val & 0x01
                MSbyte = 0
                cmd[14]=LSbyte
                cmd[15]=MSbyte
            if (tag == "STIM_LEVEL"):
                val = int(fields[1],0)
                self.logger.debug(f'STIM_LEVEL - {val}')
                LSbyte = val & 0xff
                MSbyte = 0
                cmd[16]=LSbyte
                cmd[17]=MSbyte
            if (tag == "STIM_RATE"):
                val = int(fields[1],0)
                self.logger.debug(f'STIM_RATE - {val}')
                LSbyte = val & 0xff
                MSbyte = 0
                cmd[18]=LSbyte
                cmd[19]=MSbyte
            if (tag == "EN_WR_UART"):
                val = int(fields[1],0)
                self.logger.debug(f'EN_WR_UART - {val}')
                LSbyte = val & 0x01
                MSbyte = 0
                cmd[20]=LSbyte
                cmd[21]=MSbyte
            if (tag == "FLASH_RATE"):
                val = int(fields[1],0)
                self.logger.debug(f'FLASH_RATE - {val}')
                LSbyte = val & 0x07
                MSbyte = 0
                cmd[22]=LSbyte
                cmd[23]=MSbyte
            if (tag == "FLASH_LEVEL"):
                val = int(fields[1],0)
                self.logger.debug(f'FLASH_LEVEL - {val}')
                LSbyte = val & 0x1f
                MSbyte = 0
                cmd[24]=LSbyte
                cmd[25]=MSbyte
            if (tag == "FLASH_WIDTH"):
                val = int(fields[1],0)
                self.logger.debug(f'FLASH_WIDTH - {val}')
                LSbyte = val & 0x0f
                MSbyte = 0
                cmd[26]=LSbyte
                cmd[27]=MSbyte

    # given a config dictionary, make a MAROC config command
    #
    def make_maroc_cmd(self, config: dict[str, Any], cmd: bytearray) -> None:
        cmd[0] = 0x01
        for tag, val in config.items():
            # Make a list of the should-be 4 ascii values
            vals = val.split(",")

            # Make a list of integers
            vals_int = []
            for i in range(len(vals)):
                vals_int.append(int(vals[i],0))

            # For each tag, set the appropriate bit field
            if (tag == "OTABG_ON"):
                self.set_bits_4(tag, vals_int, 0, 1)
            if (tag == "DAC_ON"):
                self.set_bits_4(tag, vals_int, 1, 1)
            if (tag == "SMALL_DAC"):
                self.set_bits_4(tag, vals_int, 2, 1)
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
            if (tag == "ENB_OUT_ADC"):
                self.set_bits_4(tag, vals_int, 23, 1)
            if (tag == "INV_START_GRAY"):
                self.set_bits_4(tag, vals_int, 24, 1)
            if (tag == "RAMP8B"):
                self.set_bits_4(tag, vals_int, 25, 1)
            if (tag == "RAMP10B"):
                self.set_bits_4(tag, vals_int, 26, 1)
            if (tag == "CMD_CK_MUX"):
                self.set_bits_4(tag, vals_int, 155, 1)
            if (tag == "D1_D2"):
                self.set_bits_4(tag, vals_int, 156, 1)
            if (tag == "INV_DISCR_ADC"):
                self.set_bits_4(tag, vals_int, 157, 1)
            if (tag == "POLAR_DISCRI"):
                self.set_bits_4(tag, vals_int, 158, 1)
            if (tag == "ENB3ST"):
                self.set_bits_4(tag, vals_int, 159, 1)
            if (tag == "VAL_DC_FSB2"):
                self.set_bits_4(tag, vals_int, 160, 1)
            if (tag == "SW_FSB2_50F"):
                self.set_bits_4(tag, vals_int, 161, 1)
            if (tag == "SW_FSB2_100F"):
                self.set_bits_4(tag, vals_int, 162, 1)
            if (tag == "SW_FSB2_100K"):
                self.set_bits_4(tag, vals_int, 163, 1)
            if (tag == "SW_FSB2_50K"):
                self.set_bits_4(tag, vals_int, 164, 1)
            if (tag == "VALID_DC_FS"):
                self.set_bits_4(tag, vals_int, 165, 1)
            if (tag == "CMD_FSB_FSU"):
                self.set_bits_4(tag, vals_int, 166, 1)
            if (tag == "SW_FSB1_50F"):
                self.set_bits_4(tag, vals_int, 167, 1)
            if (tag == "SW_FSB1_100F"):
                self.set_bits_4(tag, vals_int, 168, 1)
            if (tag == "SW_FSB1_100K"):
                self.set_bits_4(tag, vals_int, 169, 1)
            if (tag == "SW_FSB1_50k"):
                self.set_bits_4(tag, vals_int, 170, 1)
            if (tag == "SW_FSU_100K"):
                self.set_bits_4(tag, vals_int, 171, 1)
            if (tag == "SW_FSU_50K"):
                self.set_bits_4(tag, vals_int, 172, 1)
            if (tag == "SW_FSU_25K"):
                self.set_bits_4(tag, vals_int, 173, 1)
            if (tag == "SW_FSU_40F"):
                self.set_bits_4(tag, vals_int, 174, 1)
            if (tag == "SW_FSU_20F"):
                self.set_bits_4(tag, vals_int, 175, 1)
            if (tag == "H1H2_CHOICE"):
                self.set_bits_4(tag, vals_int, 176, 1)
            if (tag == "EN_ADC"):
                self.set_bits_4(tag, vals_int, 177, 1)
            if (tag == "SW_SS_1200F"):
                self.set_bits_4(tag, vals_int, 178, 1)
            if (tag == "SW_SS_600F"):
                self.set_bits_4(tag, vals_int, 179, 1)
            if (tag == "SW_SS_300F"):
                self.set_bits_4(tag, vals_int, 180, 1)
            if (tag == "ON_OFF_SS"):
                self.set_bits_4(tag, vals_int, 181, 1)
            if (tag == "SWB_BUF_2P"):
                self.set_bits_4(tag, vals_int, 182, 1)
            if (tag == "SWB_BUF_1P"):
                self.set_bits_4(tag, vals_int, 183, 1)
            if (tag == "SWB_BUF_500F"):
                self.set_bits_4(tag, vals_int, 184, 1)
            if (tag == "SWB_BUF_250F"):
                self.set_bits_4(tag, vals_int, 185, 1)
            if (tag == "CMD_FSB"):
                self.set_bits_4(tag, vals_int, 186, 1)
            if (tag == "CMD_SS"):
                self.set_bits_4(tag, vals_int, 187, 1)
            if (tag == "CMD_FSU"):
                self.set_bits_4(tag, vals_int, 188, 1)

            #Look for a MASKOR1 value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("MASKOR1"):
                chan = tag.split('_')[1]
                chan_int = int(chan)
                self.set_bits_4(tag, vals_int, 154-(2*chan_int), 1)
            #Look for a MASKOR2 value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("MASKOR2"):
                chan = tag.split('_')[1]
                chan_int = int(chan)
                self.set_bits_4(tag, vals_int, 153-(2*chan_int), 1)
            #Look for a CTEST value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("CTEST"):
                chan = tag.split('_')[1]
                chan_int = int(chan)
                #if chan in range(4):
                    #vals_int = [0,0,0,0]
                self.set_bits_4(tag, vals_int, 828-chan_int, 1)
                #print(tag, vals_int, chan)

            #Look for a GAIN value; chan is in range 0-63, with a quad of values, one for each chip
            if tag.startswith("GAIN"):
                chan = tag.split('N')[1]
                chan_int = int(chan)
                #Another list, with integer values, bits reversed
                vals_revbits = []
                for i in range (4):
                    vals_revbits.append(reverse_bits((vals_int[i]),8))
                self.set_bits_4(tag, vals_revbits, 757-9*chan_int,8)
            for ii in range(104):
                cmd[ii+4] = self.MAROC_regs[0][ii]
                cmd[ii+132] = self.MAROC_regs[1][ii]
                cmd[ii+260] = self.MAROC_regs[2][ii]
                cmd[ii+388] = self.MAROC_regs[3][ii]

    # given a config dictionary, make a trigger mask config command
    #
    def make_trigger_mask_cmd(self, config: dict[str, int], cmd: bytearray) -> None:
        for tag, val in config.items():
            if(tag.startswith('CHANMASK')):
                ch = tag.split('_')[1]
                ch_int = int(ch)
                for j in range(4): 
                    cmd[4+ch_int*4+j] = (val>>j*8) & 0xff 

    # given a config dictionary, make a goe mask config command
    #
    def make_goe_mask_cmd(self, config: dict[str, int], cmd: bytearray) -> None:
        for tag, val in config.items():
            if(tag == 'GOEMASK'):
                cmd[4] = val & 0x03

    # Set bits in MAROC_regs[chip] according to the input values.
    # Maximum value for field_width is 16 (a value can only span three bytes)
    #
    def set_bits(self, chip: int, lsb_pos: int, field_width: int, value: int) -> None:
        if (field_width >16):
            return
        if ((field_width + lsb_pos) > SERIAL_COMMAND_LENGTH):
            return
        shift = (lsb_pos % 8)
        byte_pos = int((lsb_pos+7-shift)/8)
        mask=0
        for _ii in range(0, field_width):
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
    def set_bits_4(self, tag: str, vals: list[int], lsb_pos: int, field_width: int) -> None:
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
def parse_quabo_config_file(path: str | Path) -> dict[str, str]:
    x: dict[str, str] = {}
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

def reverse_bits(data_in: int, width: int) -> int:
    data_out = 0
    for _ii in range(width):
        data_out = data_out << 1
        if (data_in & 1):
            data_out = data_out | 1
        data_in = data_in >> 1
    return data_out

# write maroc config cmd to a file
def write_maroc_config_cmd() -> None:
    q = QUABO(ip_address('1.1.1.1'))
    config = parse_quabo_config_file('quabo_config.txt')
    cmd = bytearray(492)
    q.make_maroc_cmd(config, cmd)
    with open('maroc_cmd_new.bin', 'wb') as f:
        f.write(cmd)


# parse the data config file to get DAQ params for quabos
#
def get_daq_params(data_config: DataConfig) -> DAQ_PARAMS:
    """Translate the high-level data configuration into Quabo-level DAQ parameters.
    
    Parses image mode settings (integration time, sample size), pulse-height 
    mode settings (any_trigger, grouping), and test signals (flash/stim).

    Args:
        data_config: The validated science/engineering configuration model.

    Returns:
        An initialized quabo_driver.DAQ_PARAMS object.
    """
    do_image = False
    image_usec = 1
    image_8bit = False
    do_ph = False
    bl_subtract = True
    do_any_trigger = False
    group_ph_frames = False
    if data_config.image:
        do_image = True
        image = data_config.image
        if image.quabo_sample_size == 8:
            image_8bit = True
        image_usec = image.integration_time_usec
    if data_config.pulse_height:
        do_ph = True
        if data_config.pulse_height.any_trigger:
            do_any_trigger = True
            any_trigger = data_config.pulse_height.any_trigger
            if any_trigger.group_ph_frames == 1:
                group_ph_frames = True
    daq_params = DAQ_PARAMS(
        do_image, image_usec - 1, image_8bit, do_ph, bl_subtract, do_any_trigger, group_ph_frames
    )
    if data_config.flash_params:
        fp = data_config.flash_params
        daq_params.set_flash_params(fp.rate, fp.level, fp.width)
    if data_config.stim_params:
        sp = data_config.stim_params
        daq_params.set_stim_params(sp.rate, sp.level)
    return daq_params
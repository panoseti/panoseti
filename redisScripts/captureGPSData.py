import time
import serial
import struct
import redis
from signal import signal, SIGINT

BYTEORDER = 'big'
RKEY = 'GPSRECEIVERPRIM'
RKEYsupp = 'GPSRECEIVERSUPP'

r = redis.Redis(host='localhost', port=6379, db=0)

def handler(signal_recieved, frame):
    print('\nSIGINT or CTRL-C detected. Exiting')
    exit(0)

def floatfrom_bytes(bytesData, bytesorder=BYTEORDER):
    if BYTEORDER == 'little':
        f = '<f'
    else:
        f = '>f'

    return struct.unpack(f, bytesData)[0]

def doublefrom_bytes(bytesData, bytesorder=BYTEORDER):
    if BYTEORDER == 'little':
        d = '<d'
    else:
        d = '>d'

    return struct.unpack(d, bytesData)[0]

timingFlagValues = {0:'GPS', 1:'UTC'}

# OutputID 0x8F-AB
def primaryTimingPacket(data):
    
    timeofWeek = int.from_bytes(data[1:5], byteorder=BYTEORDER, signed=False)
    
    weekNumber = int.from_bytes(data[5:7], byteorder=BYTEORDER, signed=False)
    
    UTCOffset = int.from_bytes(data[7:9], byteorder=BYTEORDER, signed=True)
    
    timingFlag = int.from_bytes(data[9:10], byteorder=BYTEORDER, signed=False)
    time = timingFlag & 0x01
    PPS = (timingFlag & 0x02) >> 1
    timeSet = (timingFlag & 0x04) >> 2
    UTCinfo = (timingFlag & 0x08) >> 3
    timeFrom = (timingFlag & 0x10) >> 4
    
    seconds = int.from_bytes(data[10:11], byteorder=BYTEORDER, signed=False)
    minutes = int.from_bytes(data[11:12], byteorder=BYTEORDER, signed=False)
    hours = int.from_bytes(data[12:13], byteorder=BYTEORDER, signed=False)
    dayofMonth = int.from_bytes(data[13:14], byteorder=BYTEORDER, signed=False)
    month = int.from_bytes(data[14:15], byteorder=BYTEORDER, signed=False)
    year = int.from_bytes(data[15:17], byteorder=BYTEORDER, signed=False)
    
    
    r.hset(RKEY,'TOW', timeofWeek)
    r.hset(RKEY, 'WEEKNUMBER', weekNumber)
    r.hset(RKEY, 'UTCOFFSET', UTCOffset)
    r.hset(RKEY, 'GPSTIME', str(year)+'_'+str(month)+'_'+str(dayofMonth)+'T'+str(hours)+':'+str(minutes)+':'+str(seconds))
    
    #print('Time of Week = ', timeofWeek)
    #print('Week number = ', weekNumber)
    #print('UTC offset = ', UTCOffset)
    #print('Timing Flag = ', timingFlag)
    r.hset(RKEY, 'TIMEFLAG', timingFlagValues[time])
    r.hset(RKEY, 'PPSFLAG', timingFlagValues[PPS])
    if timeSet == 0:
        r.hset(RKEY, 'TIMESETFLAG', 'time is set')
    else:
        r.hset(RKEY, 'TIMESETFLAG', 'time is not set')
    
    if UTCinfo == 0:
        r.hset(RKEY, 'UTCINFO', 'have UTC info')
    else:
        r.hset(RKEY, 'UTCINFO', 'no UTC info')
        
    if timeFrom == 0:
        r.hset(RKEY, 'TIMEFROMFLAG', 'time from GPS')
    else:
        r.hset(RKEY, 'TIMEFROMFLAG', 'time from user')
    
    #print(str(year)+'_'+str(month)+'_'+str(dayofMonth)+'T'+str(hours)+':'+str(minutes)+':'+str(seconds))
    

    
recModeValues = {0:'Automatic (2D/3D)', 1:'Single Satellite (Time)', 3:'Horizontal (2D)', 4:'Full Position (3D)', 7:'Over-Determined Clock'}
disModeValues = {0:'Normal (Locked to GPS)', 1:'Power Up', 2:'Auto Holdover', 3:'Manual Holdover', 4:'Recovery', 5:'Not used', 6:'Disciplining Disabled'}
GPSDecodeValues = {0:'Doing fixes', 1:'Don\'t have GPS time', 3:'PDOP is too high', 
                   8:'No usable sats', 9:'Only 1 usable sat', 10:'Only 2 usable sats', 11:'Only 3 usable sats', 12:'The chosen sat is unusable', 
                   16:'TRAIM rejected the fix'}
disActivityValues = {0:'Phase locking', 1:'Oscillator warm-up', 2:'Frequency locking', 3:'Placing PPS', 4:'Initializing loop filter', 
                     5:'Compensating OCXO (holdover)', 6:'Inactive', 7:'Not used', 8:'Recovery mode', 9:'Calibration/control voltage'}
DEFAULTVALUE = 'Uknown Value {0}'
# OutputID 0x8F-AC
def supplimentaryTimingPacket(data):
    receiverMode = int.from_bytes(data[1:2], byteorder=BYTEORDER, signed=False)
    discipliningMode = int.from_bytes(data[2:3], byteorder=BYTEORDER, signed=False)
    selfSurveyProgress = int.from_bytes(data[3:4], byteorder=BYTEORDER, signed=False)
    holdOverDuration = int.from_bytes(data[4:8], byteorder=BYTEORDER, signed=False)
    criticalAlarms = int.from_bytes(data[8:10], byteorder=BYTEORDER, signed=False)
    minorAlarms = int.from_bytes(data[10:12], byteorder=BYTEORDER, signed=False)
    GPSDecodingStatus = int.from_bytes(data[12:13], byteorder=BYTEORDER, signed=False)
    discipliningActivity = int.from_bytes(data[13:14], byteorder=BYTEORDER, signed=False)
    spareStatus1 = int.from_bytes(data[14:15], byteorder=BYTEORDER, signed=False)
    spareStatus2 = int.from_bytes(data[15:16], byteorder=BYTEORDER, signed=False)

    PPSOffset = floatfrom_bytes(data[16:20])
    clockOffset = floatfrom_bytes(data[20:24])
    DACValue = int.from_bytes(data[24:28], byteorder=BYTEORDER, signed=False)
    DACVoltage = floatfrom_bytes(data[28:32])
    temp = floatfrom_bytes(data[32:36])
    latitude = doublefrom_bytes(data[36:44])
    longitude = doublefrom_bytes(data[44:52])
    altitude = doublefrom_bytes(data[52:60])
    PPSQuantizationError = floatfrom_bytes(data[60:64])

    
    if receiverMode in recModeValues:
        r.hset(RKEYsupp, 'RECEIVERMODE', recModeValues[receiverMode])
    else:
        r.hset(RKEYsupp, 'RECEIVERMODE', DEFAULTVALUE.format(receiverMode))
    
    if discipliningMode in disModeValues:
        r.hset(RKEYsupp, 'DISCIPLININGMODE', disModeValues[discipliningMode])
    else:
        r.hset(RKEYsupp, 'DISCIPLININGMODE', DEFAULTVALUE.format(disModeValues))
    
    r.hset(RKEYsupp, 'SELFSURVEYPROGRESS', selfSurveyProgress)
    r.hset(RKEYsupp, 'HOLDOVERDURATION', holdOverDuration)
    #ALARMS
    r.hset(RKEYsupp, 'DACatRail', (criticalAlarms & 0x08) >> 3)
    
    r.hset(RKEYsupp, 'DACnearRail', minorAlarms & 0x0001)
    r.hset(RKEYsupp, 'AntennaOpen', (minorAlarms & 0x0002) >> 1)
    r.hset(RKEYsupp, 'AntennaShorted', (minorAlarms & 0x0004) >> 2)
    r.hset(RKEYsupp, 'NotTrackingSatellites', (minorAlarms & 0x0008) >> 3)
    r.hset(RKEYsupp, 'NotDiscipliningOscillator', (minorAlarms & 0x0010) >> 4)
    r.hset(RKEYsupp, 'SurveyInProgress', (minorAlarms & 0x0020) >> 5)
    r.hset(RKEYsupp, 'NoStoredPosition', (minorAlarms & 0x0040) >> 6)
    r.hset(RKEYsupp, 'LeapSecondPending', (minorAlarms & 0x0080) >> 7)
    r.hset(RKEYsupp, 'InTestMode', (minorAlarms & 0x0100) >> 8)
    r.hset(RKEYsupp, 'PositionIsQuestionable', (minorAlarms & 0x0200) >> 9)
    r.hset(RKEYsupp, 'EEPROMCorrupt', (minorAlarms & 0x0400) >> 10)
    r.hset(RKEYsupp, 'AlmanacNotComplete', (minorAlarms & 0x0800) >> 11)
    r.hset(RKEYsupp, 'PPSNotGenerated', (minorAlarms & 0x1000) >> 12)
    
    if GPSDecodingStatus in GPSDecodeValues:
        r.hset(RKEYsupp, 'GPSDECODINGSTATUS', GPSDecodeValues[GPSDecodingStatus])
    else:
        r.hset(RKEYsupp, 'GPSDECODINGSTATUS', DEFAULTVALUE.format(GPSDecodingStatus))
        
    if discipliningActivity in disActivityValues:
        r.hset(RKEYsupp, 'DISCIPLININGACTIVITY', disActivityValues[discipliningActivity])
    else:
        r.hset(RKEYsupp, 'DISCIPLININGACTIVITY', DEFAULTVALUE.format(discipliningActivity))
    
    r.hset(RKEYsupp, 'SPARESTATUS1', spareStatus1)
    r.hset(RKEYsupp, 'SPARESTATUS2', spareStatus2)
    
    r.hset(RKEYsupp, 'PPSOFFSET', PPSOffset)
    r.hset(RKEYsupp, 'CLOCKOFFSET', clockOffset)
    r.hset(RKEYsupp, 'DACVALUE', DACValue)
    r.hset(RKEYsupp, 'DACVOLTAGE', DACVoltage)
    r.hset(RKEYsupp, 'TEMPERATURE', temp)
    r.hset(RKEYsupp, 'LATITUDE', latitude)
    r.hset(RKEYsupp, 'LONGITUDE', longitude)
    r.hset(RKEYsupp, 'ALTITUDE', altitude)
    r.hset(RKEYsupp, 'PPSQUANTIZATIONERROR', PPSQuantizationError)
    
    #print('ID = ', data[0:1])
    #print('ReceiverMode = ', receiverMode, ' bytes = ', data[1:2])
    #print('DiscipliningMode = ', discipliningMode, ' bytes = ', data[2:3])
    #print('Self-SurveyProgress = ', selfSurveyProgress, ' bytes = ', data[3:4])
    #print('Holdover Duration = ', holdOverDuration, ' bytes = ', data[4:8])
    #print('Critical Alarms = ', criticalAlarms, ' bytes = ', data[8:10])
    #print('Minor Alarms = ', minorAlarms, ' bytes = ', data[10:12])
    #print('GPS Decoding Status = ', GPSDecodingStatus, ' bytes = ', data[12:13])
    #print('Disciplining Activity = ', discipliningActivity, ' bytes = ', data[13:14])
    #print('Spare Status1 = ', spareStatus1, ' bytes = ', data[14:15])
    #print('Spare Status2 = ', spareStatus2, ' bytes = ', data[15:16])
    #print('PPSOffset = ', PPSOffset, ' bytes = ', data[16:20])
    #print('Clock Offset = ', clockOffset, ' bytes = ', data[20:24])
    #print('DAC Values = ', DACValue, ' bytes = ', data[24:28])
    #print('DAC Voltage = ', DACVoltage, ' bytes = ', data[28:32])
    #print('Temperature = ', temp, ' bytes = ', data[32:36])
    #print('Lat = ', latitude, ' Long = ', longitude, ' Altitude = ', altitude)
    #print('PPS Quantization Error = ', PPSQuantizationError, ' bytes = ', data[60:64])



# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate=9600,
    timeout=1,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)
ser.isOpen()
# Reading the data from the serial port. This will be running in an infinite loop.

signal(SIGINT, handler)
data = b''
dataSize = 0
bytesToRead = 0

print('Running')
while 1 :
    while bytesToRead == 0:
        bytesToRead = ser.inWaiting()
    data += ser.read(bytesToRead)
    dataSize += bytesToRead
    if data[dataSize-1:dataSize] == b'\x03' and data[dataSize-2:dataSize-1] == b'\x10':
        if data[0:1] == b'\x10':
            id = data[1:3]
            if id == b'\x8f\xab':
                primaryTimingPacket(data[2:dataSize-2])
                r.hset('UPDATED', RKEY, 1)
            elif id == b'\x8f\xac':
                supplimentaryTimingPacket(data[2:dataSize-2])
                r.hset('UPDATED', RKEYsupp, 1)
        
        print(data[1:3])
        data = b''
        dataSize = 0
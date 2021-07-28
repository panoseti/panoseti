import redis
import time
from signal import signal, SIGINT
from datetime import datetime
#from panosetiSIconvert import HKconvert
# HKconv = HKconvert()

def handler(signal_recieved, frame):
    print('\nSIGINT or CTRL-C detected. Exiting')
    exit(0)
signal(SIGINT, handler)

# Intializing Global Constants
TITLE = 'REDIS SNAPSHOT'
FONTSIZE = '15'
UPPERCOLOR = 'red'
LOWERCOLOR = 'orange'
NORMALCOLOR = 'green'
TEXTCOLOR = 'black'
REDISIP = 'localhost'
REDISPORT = 6379

#Starting Redis connection
r = redis.Redis(host=REDISIP, port=REDISPORT, db=0)

#Defining the Quabo Bound for the upper and lower threshold of values.
quaboBounds = {b'HVMON0': [63,66,0], #Each Key has 3 values [Lower Threshold, Upper Threshold, FlipBit] If FlipBit is 1 then Upper becomes lower threshold
             b'HVMON1': [10,20,0],
             b'HVMON2': [50,60,0],
             b'HVMON3': [0,1,0],
             b'HVIMON0': [0,1,0],
             b'HVIMON1': [0,1,0],
             b'HVIMON2': [0,1,1],
             b'HVIMON3': [0,1,0],
             b'RAWHVMON': [0,1,0],
             b'V12MON': [0,1,0],
             b'V18MON': [0,1,0],
             b'V33MON': [0,1,0],
             b'V37MON': [0,1,0],
             b'I10MON': [0,1,0],
             b'I18MON': [0,1,0],
             b'I33MON': [0,1,0],
             b'TEMP1': [0,1,1],
             b'TEMP2': [0,1,0],
             b'VCCINT': [0,1,0],
             b'VCCAUX': [0,1,0],
             b'SHUTTER_STATUS': [0,1,0],
             b'LIGHT_SENSOR_STATUS': [0,1,0],
             b'FWID0': [0,1,0],
             b'FWID1': [0,1,0],
             b'StartUp': [0,1,0]}

#Initalizing Template Values.
htmlTemplate = '''<html>\n<head><style> \
table, th, td {{
  border: 1px solid black;
  padding: 5px;
}}
table {{
  border-spacing: 15px;
}}
</style></head>\n<body>\n{0}\n</body>\n</html>'''
mainHeader = '<h1 style="font-size:{1}px">{0}</h1>\n'.format(TITLE, FONTSIZE)
subHeader = '<h2 style="font-size:{1}px">{0}</h2>\n'
tableHeader = '<table style="Width:100%;font-size:{1}px">\n<tbody>\n{0}\n</tbody>\n</table>'
rowHeader = '<tr>\n{0}</tr>'
eleTemplate = '<td style="color:{1};">{0}</td>\n'

def determineColor(value, bounds):
    if value > bounds[0]:
        if value > bounds[1]:
            if bounds[2]:
                return NORMALCOLOR
            else:
                return UPPERCOLOR
        else:
            return LOWERCOLOR
    else:
        if bounds[2]:
            return UPPERCOLOR
        else:
            return NORMALCOLOR


labels = ''
quaboKeys = []
redisKeys = [int(i.decode("utf-8")) for i in r.keys("[0-9]*")]
redisKeys.sort()
recieverKeys = [i.decode("utf-8") for i in r.keys("[A-Z]*")]
recieverKeys.remove('UPDATED')
#print(recieverKeys)

for k in r.hkeys(redisKeys[0]):
    quaboKeys.append(k)
    labels += eleTemplate.format(k.decode("utf-8"), TEXTCOLOR)

while 1:
    htmlBody = mainHeader
    
    rows = rowHeader.format(labels)
    
    for q in redisKeys:
        elements = ''
        setValues = r.hgetall(q)
        for k in quaboKeys:
            #print(k)
            currValue = setValues[k].decode("utf-8")#HKconv.convertValue(k.decode("utf-8"), setValues[k].decode("utf-8"))
            color = TEXTCOLOR
            if k in quaboBounds:
                color = determineColor(float(currValue), quaboBounds[k])
            elements += eleTemplate.format(currValue, color)
        rows += rowHeader.format(elements)
    
    htmlBody += subHeader.format('Quabos', FONTSIZE)
    htmlBody += tableHeader.format(rows, FONTSIZE)
    
    for key in recieverKeys:
        label = ''
        value = ''
        setValues = r.hgetall(key)
        for k in setValues.keys():
            label += eleTemplate.format(k.decode("utf-8"), TEXTCOLOR)
            value += eleTemplate.format(setValues[k].decode("utf-8"), TEXTCOLOR)
        rows = rowHeader.format(label) +  rowHeader.format(value)
        htmlBody += subHeader.format(key, FONTSIZE)
        htmlBody += tableHeader.format(rows, FONTSIZE)
        
    f = open("index.html", "w")
    f.write(htmlTemplate.format(htmlBody))
    f.close()
    print(datetime.utcnow())
    time.sleep(1)

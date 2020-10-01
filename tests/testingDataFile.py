import h5py
import sys

def compare(a,b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def checkIMG16Data(IMG16):
    if len(IMG16) == 0:
        print("Empty 16bit Image Data")
        return False
    refData = [0]*256
    index = 0
    mode = 0
    pktNum = 0
    ntp_sec = -1
    for name in IMG16:
        data = IMG16[name]
        ntp_usec = -1
        for i in range(8):
            if (not compare(IMG16[name][i], refData)):
                print("Error in Dataset ", name)
                print("Expected Image Values are ", refData)
                print("Actual Image Values are ", IMG16[name][i])
                return False
            if pktNum != IMG16[name].attrs['PKTNUM'][i]:
                print("Error in Dataset ", name)
                print("Expected PktNum Values are ", pktNum)
                print("Actual PktNum Values are ", IMG16[name].attrs['PKTNUM'])
                return False
            if ntp_sec > IMG16[name].attrs['ntp_sec'][i]:
                print("Error in Dataset ", name)
                print("Error in ntp_sec")
                print(list(IMG16[name].attrs.items()))
                return False
            ntp_sec = IMG16[name].attrs['ntp_sec'][i]
            if ntp_usec > IMG16[name].attrs['ntp_usec'][i]:
                print("Error in Dataset ", name)
                print("Error in ntp_usec")
                print(list(IMG16[name].attrs.items()))
                return False
            ntp_usec = IMG16[name].attrs['ntp_usec'][i]
            
        pktNum+=1
        mode = index // 256
        if mode == 0:
            refData[index%256] = 1
        elif mode == 1:
            refData[index%256] = 257
        elif mode == 2:
            refData[index%256] = 65535
        else:
            refData = [0]*256
        index += 1
    return True

def checkIMG8Data(IMG8):
    if len(IMG8) == 0:
        print("Empty 8bit Image Data")
        return False
    refData = [0]*256
    index = 0
    mode = 0
    pktNum = 0
    ntp_sec = -1
    for name in IMG8:
        data = IMG8[name]
        ntp_usec = -1
        for i in range(8):
            if (not compare(IMG8[name][i], refData)):
                print("Error in Dataset ", name)
                print("Expected Image Values are ", refData)
                print("Actual Image Values are ", IMG8[name][i])
                return False
            if pktNum != IMG16[name].attrs['PKTNUM'][i] and IMG16[name].attrs['PKTNUM'][i] != 0:
                print("Error in Dataset ", name)
                print("Expected PktNum Values are ", pktNum)
                print("Actual PktNum Values are ", IMG8[name].attrs['PKTNUM'])
                return False
            if ntp_sec > IMG8[name].attrs['ntp_sec'][i] and IMG8[name].attrs['ntp_sec'][i] != 0:
                print("Error in Dataset ", name)
                print("Error in ntp_sec")
                print(list(IMG8[name].attrs.items()))
                return False
            ntp_sec = IMG8[name].attrs['ntp_sec'][i]
            if ntp_usec > IMG8[name].attrs['ntp_usec'][i] and IMG8[name].attrs['ntp_usec'][i] != 0:
                print("Error in Dataset ", name)
                print("Error in ntp_usec")
                print(list(IMG8[name].attrs.items()))
                return False
            ntp_usec = IMG8[name].attrs['ntp_usec'][i]
        
        pktNum +=1
        mode = index // 256
        if mode == 0:
            refData[index%256] = 1
        elif mode == 1:
            refData[index%256] = 255
        else:
            refData = [0]*256
        index += 1
    return True

def checkPHData(PHData):
    if len(PHData) == 0:
        print("Empty PH Data")
        return False
    refData = [0]*256
    index = 0
    mode = 0
    pktNum = 0
    ntp_sec = -1
    for name in PHData:
        data = PHData[name]
        ntp_usec = -1
        if (not compare(PHData[name], refData)):
            print("Error in Dataset ", name)
            print("Expected Image Values are ", refData)
            print("Actual Image Values are ", PHData[name])
            return False
        if pktNum != PHData[name].attrs['PKTNUM']:
            print("Error in Dataset ", name)
            print("Expected PktNum Values are ", pktNum)
            print("Actual PktNum Values are ", PHData[name].attrs['PKTNUM'])
            return False
        if ntp_sec > PHData[name].attrs['ntp_sec']:
            print("Error in Dataset ", name)
            print("Error in ntp_sec")
            print(list(PHData[name].attrs.items()))
            return False
        ntp_sec = PHData[name].attrs['ntp_sec']
        if ntp_usec > PHData[name].attrs['ntp_usec']:
            print("Error in Dataset ", name)
            print("Error in ntp_usec")
            print(list(PHData[name].attrs.items()))
            return False
        ntp_usec = PHData[name].attrs['ntp_usec']
        
        pktNum+=1
        mode = index // 256
        if mode == 0:
            refData[index%256] = 1
        elif mode == 1:
            refData[index%256] = 257
        elif mode == 2:
            refData[index%256] = 65535
        index += 1
    return True

def checkHKPackets(DynamicData):
    if len(DynamicData) == 0:
        print("Empty HouseKeeping Data")
        return False
    dataRef = [[0]*26, [257]*26, [514]*26, [771]*26]
    dataRef[1][21] = 72340172838076673
    dataRef[1][22] = 1
    dataRef[1][23] = 0
    dataRef[1][24] = 16843009
    dataRef[1][25] = 16843009
    
    dataRef[2][21] = 144680345676153346
    dataRef[2][22] = 0
    dataRef[2][23] = 1
    dataRef[2][24] = 33686018
    dataRef[2][25] = 33686018
    
    dataRef[3][21] = 217020518514230019
    dataRef[3][22] = 1
    dataRef[3][23] = 1
    dataRef[3][24] = 50529027
    dataRef[3][25] = 50529027

    for i in DynamicData:
        if len(DynamicData[i]) == 0:
            print("Empty HouseKeeping Data")
            return False
        for j in range(len(DynamicData[i])):
            for k in range(2, len(DynamicData[i][j])):
                if DynamicData[i][j][k] != dataRef[j][k-1]:
                    print("Error In HouseKeeping Data")
                    print("Table Name ", i)
                    print("Expected Values are ", dataRef[j])
                    print("Actual Values are ", DynamicData[i][j])
                    return False
    return True


if len(sys.argv) != 2:
    print("Program to run test on packet output from packetTestGenerator.")
    print("Please provide a data file to be tested on")
    exit(0)

fileName = sys.argv[1]

f = h5py.File(fileName)

IMG16 = f['bit16IMGData']['ModulePair_00254_00001']
IMG8 = f['bit8IMGData']['ModulePair_00254_00001']
PHData = f['PHData']
DynamicData = f['DynamicMeta']['ModulePair_00254_00001']

print("Testing 16 bit Image Data")
if checkIMG16Data(IMG16):
    print("Test passed")
else:
    print("Test Failed")
    exit(0)

print("Testing PH Data")
if checkPHData(PHData):
    print("Test passed")
else:
    print("Test Failed")
    exit(0)

print("Testing 8 bit Image Data")
if checkIMG8Data(IMG8):
    print("Test passed")
else:
    print("Test Failed")
    exit(0)

print("Testing House Keeping Data")
if checkHKPackets(DynamicData):
    print("Test passed")
else:
    print("Test Failed")
    exit(0)

f.close()


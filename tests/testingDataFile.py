import h5py
import sys

def compare(a,b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            print("Index ", i, " is incorrect.")
            return False
    return True

def printError(name, refData, IMGData):
    print("Error in Dataset ", name)
    print("Expected Image Values are ")
    for j in range(16):
        print(j*16, refData[j*16:j*16+16])
    print("Actual Image Values are ")
    for j in range(16):
        print(j*16, IMGData[j*16:j*16+16])

def checkIMGData(IMGData, bitMode=16, PHMode=False):
    if len(IMGData) == 0:
        print("Empty 16bit Image Data")
        return False
    refData = [0]*256
    index = 0
    
    datasetSize = len(IMGData)//6
    
    for datasetIndex in range(datasetSize):
        dataset = IMGData["DATA{0:09d}".format(datasetIndex)]
        for framePair in dataset:
            for frame in framePair:
                if not compare(refData, frame):
                    print("Error on framePair ", index)
                    printError("DATA{0:09d}".format(datasetIndex), refData, frame)
                    return False
            mode = index // 256
            if bitMode == 16:
                if mode == 0:
                    refData[index%256] = 1
                elif mode == 1:
                    refData[index%256] = 257
                elif mode == 2:
                    refData[index%256] = 65535
                else:
                    break
            elif bitMode == 8:
                if mode == 0:
                    refData[index%256] = 1
                elif mode == 1:
                    refData[index%256] = 255
                else:
                    break
            index += 1
        
        dataset = IMGData["DATA{0:09d}_pktNum".format(datasetIndex)]
        pktNum = 0
        for framePair in dataset:
            for frame in framePair:
                if frame[0] != pktNum:
                    print("pktNum is supposed to be ", pktNum, " but it is ", frame[0])
                    return False
            pktNum += 1
            if pktNum >= index:
                break
        
        if PHMode:
            return True
            
        count = 0
        for statusIndex in range(index):
            dataset = IMGData["DATA{0:09d}_status".format(datasetIndex)]
            for framePair in dataset:
                if framePair[0][0] != 255:
                    print("Count is ", count)
                    print("Status bit at ", statusIndex, "is ", framePair[0][0])
                    return False
                count += 1
                if count >= index:
                    break
        
                    
    return True

def checkHKPackets(DynamicData):
    if len(DynamicData) == 0:
        print("Empty HouseKeeping Data")
        return False
    dataRef = [[0]*26, [257]*26, [514]*26, [771]*26, [0]*26]
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

    dataRef[4][17] = -1
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

f = h5py.File(fileName, 'r')

IMG16 = f['bit16IMGData']['ModulePair_00254_00001']
IMG8 = f['bit8IMGData']['ModulePair_00254_00001']
PHData = f['PHData']['ModulePair_00254_00001']
DynamicData = f['DynamicMeta']
print("Testing 16 bit Image Data")
if checkIMGData(IMG16, bitMode=16):
    print("Test passed")
else:
    print("Test Failed")
    exit(0)

print("Testing 8 bit Image Data")
if checkIMGData(IMG8, bitMode=8):
    print("Test passed")
else:
    print("Test Failed")
    exit(0)

print("Testing PH Data")    
if checkIMGData(PHData, bitMode=16, PHMode=True):
    print("Test passed")
else:
    print("Test Failed")
    exit(0)

#print("Testing House Keeping Data")
#if checkHKPackets(DynamicData):
#    print("Test passed")
#else:
#    print("Test Failed")
#    exit(0)

f.close()


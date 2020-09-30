import h5py
import sys

def statusFreq(IMG):
    d = {}
    for i in IMG:
        stat = IMG[i].attrs['status']
        if stat in d:
            d[stat] += 1
        else:
            d[stat] = 0
    print(d)

if len(sys.argv) != 2:
    print("Program to get the frequency of status bit for a file")
    print("Please provide a data file to be tested on")
    exit(0)

fileName = sys.argv[1]

f = h5py.File(fileName)

IMG16 = f['bit16IMGData']['ModulePair_00254_00001']
IMG8 = f['bit8IMGData']['ModulePair_00254_00001']

print("Status Frequency for 16 bit Image Data")
statusFreq(IMG16)
print("Status Frequency for 8 bit Image Data")
statusFreq(IMG8)

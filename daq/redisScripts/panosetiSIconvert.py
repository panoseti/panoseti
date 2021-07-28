import re

convertValues = {r'[A-Z]':1e9, r'm[A-Z]':1e6, r'u[A-Z]':1e3, r'n[A-Z]':1 }

class HKconvert():
    def __init__(self):
        self.keyFormat = {r'HVMON[0-3]':self.HVMON,
                         r'HVIMON[0-3]':self.HVIMON,
                         r'RAWHVMON':self.RAWHVMON,
                         r'V1[0-9]MON':self.V12MON,
                         r'V3[0-9]MON':self.V33MON,
                         r'I10MON':self.I10MON,
                         r'I18MON':self.I18MON,
                         r'I33MON':self.I33MON,
                         r'TEMP1':self.TEMP1,
                         r'TEMP2':self.TEMP2,
                         r'VCC*':self.VCC}
        self.voltageFactor = 1e3
        self.currentFactor = 1e3
        
    def HVMON(self, value):
        return value*1.22*1e6 / self.voltageFactor

    def HVIMON(self, value):
        return (65535-value)*38.1 / self.currentFactor

    def RAWHVMON(self, value):
        return value*1.22*1e6 / self.voltageFactor

    def V12MON(self, value):
        return value*19.07*1e3 / self.voltageFactor

    def V18MON(self, value):
        return self.V12MON(value)

    def V33MON(self, value):
        return value*38.1*1e3 / self.voltageFactor

    def V37MON(self, value):
        return self.V33MON(value)

    def I10MON(self, value):
        return value*182*1e3 / self.currentFactor

    def I18MON(self, value):
        return value*37.8*1e3 / self.currentFactor

    def I33MON(self, value):
        return self.I18MON(value)

    def TEMP1(self, value):
        return value*0.0625
    
    def TEMP2(self, value):
        return (value/130.04) - 273.15

    def VCC(self, value):
        return value*3/65536*1e9 / self.voltageFactor
        
    def showUnits(self, output=2):
        returnVal = ['','']
        if self.voltageFactor == 1e9:
            returnVal[0] = 'Volts'
        elif self.voltageFactor == 1e6:
            returnVal[0] = 'miliVolts'
        elif self.voltageFactor == 1e3:
            returnVal[0] = 'microVolts'
        else:
            returnVal[0] = 'nanoVolts'
            
        if self.currentFactor == 1e9:
            returnVal[1] = 'Amps'
        elif self.currentFactor == 1e6:
            returnVal[1] = 'miliAmps'
        elif self.currentFactor == 1e3:
            returnVal[1] = 'microAmps'
        else:
            returnVal[1] = 'nanoAmps'
        
        returnVal = ['The units for electrical potential is now '+returnVal[0], 'The units for electrical current is now '+returnVal[1]]
        
        if output == len(returnVal):
            for i in range(output):
                print(returnVal[i])
        else:
            print(returnVal[output])
        return returnVal
        
    def changeUnits(self, inputVal):
        for k in convertValues:
            if re.match(k, inputVal):
                if inputVal[-1] == 'V':
                    self.voltageFactor = convertValues[k]
                    self.showUnits(0)
                elif inputVal[-1] == 'A':
                    self.currentFactor = convertValues[k]
                    self.showUnits(1)
                return
        self.showUnits()
        return
    

        
    def convertValue(self, key, value):
        for k in self.keyFormat:
            if re.match(k, key):
                return self.keyFormat[k](int(value))
        return value
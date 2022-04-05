##############################################################
# Conversion Library for storing values into redis at different 
# units default is to store the values as Volts and Amps. The 
# units can be changed by running changeUnits(). This should be
# imported for necessary scripts and running this scripts will 
# run the unit tests for the functions
##############################################################
import re
import unittest

from numpy import place

convertValues = {r'[A-Z]':1e9, r'm[A-Z]':1e6, r'u[A-Z]':1e3, r'n[A-Z]':1 }

class HKconvert():
    def __init__(self):
        self.keyFormat = {r'HVMON[0-3]':self.HVMON,
                         r'HVIMON[0-3]':self.HVIMON,
                         r'RAWHVMON':self.RAWHVMON,
                         r'V12MON':self.V12MON,
                         r'V18MON':self.V18MON,
                         r'V3[0-9]MON':self.V33MON,
                         r'I10MON':self.I10MON,
                         r'I18MON':self.I18MON,
                         r'I33MON':self.I33MON,
                         r'TEMP1':self.TEMP1,
                         r'TEMP2':self.TEMP2,
                         r'VCC*':self.VCC}
        self.voltageFactor = 1e9
        self.currentFactor = 1e9
        
    def HVMON(self, value):
        return -value*1.22*1e6 / self.voltageFactor

    def HVIMON(self, value):
        return (65535-value)*38.1 / self.currentFactor

    def RAWHVMON(self, value):
        return -value*1.22*1e6 / self.voltageFactor

    def V12MON(self, value):
        return value*19.07*1e3 / self.voltageFactor

    def V18MON(self, value):
        return value*38.14*1e3 / self.voltageFactor

    def V33MON(self, value):
        return value*76.2*1e3 / self.voltageFactor

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
        return None

class TestHKConvert(unittest.TestCase):
    hk_converter = HKconvert()
    def test_HVMON_conversion(self):
        for i in range(4):
            # Test zero value
            self.assertEqual(0, self.hk_converter.convertValue(f"HVMON{i}", 0x0000))
            # Test increment value
            self.assertEqual(-0.00122, self.hk_converter.convertValue(f"HVMON{i}", 0x0001))
            # Test maximum value
            self.assertAlmostEqual(-80, self.hk_converter.convertValue(f"HVMON{i}", 0xffff), places=1)

    def test_HVIMON_conversion(self):
        for i in range(4):
            # Test zero value
            self.assertEqual(0, self.hk_converter.convertValue(f"HVIMON{i}", 0x0000))
            # Test increment value
            self.assertEqual(-0.00122, self.hk_converter.convertValue(f"HVIMON{i}", 0x0001))
            # Test maximum value
            self.assertAlmostEqual(-80, self.hk_converter.convertValue(f"HVIMON{i}", 0xffff), places=1)

    def test_RAWHVMON_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("RAWHVMON", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

    def test_V12MON_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("V12MON", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

    def test_V18MON_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("V18MON", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

    def test_V33MON_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("V33MON", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

    def test_V37MON_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("V37MON", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

    def test_I10MON_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("I10MON", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

    def test_I18MON_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("I18MON", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

    def test_I33MON_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("I33MON", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

    def test_TEMP1_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("TEMP1", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

    def test_TEMP2_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("TEMP2", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

    def test_VCC_conversion(self):
        # Test minimum value
        self.assertEqual(0, self.hk_converter.convertValue("VCC", 0))
        # Test maximum value
        self.assertEqual(0, 0)
        # Test standard value
        self.assertEqual(0, 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)

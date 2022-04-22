#!/usr/bin/python3

import os
import sys
import netsnmp

os.environ['MIBDIRS']='+./'

wrsSnmpObjs={'sfppn'         : 'WR-SWITCH-MIB::wrsPortStatusSfpPN' , \
            'linkstatus'    : 'WR-SWITCH-MIB::wrsPortStatusLink'  , \
            'pllstatus'     : 'WR-SWITCH-MIB::wrsSoftPLLStatus'         }

class snmp_wapper(object):
    def __init__(self, dev, obj):
        self.dev = dev
        self.obj = obj
    def snmpwalk(self):
        oid = netsnmp.Varbind(self.obj)
        try:
            res = netsnmp.snmpwalk(oid, Version=2, DestHost=self.dev, Community='public')
            if(len(res) == 0):
                return 0
            r = []
            for i in range(len(res)):
                r.append(bytes.decode(res[i]))
            return r
        except:
            return -1

class wrs_snmp(object):    
    def __init__(self, dev = '10.0.1.36', objs = wrsSnmpObjs):
        self.dev = dev
        self.objs = objs
        self.__InitMethods()
    # create methods base on the dict--wrsSnmpObj 
    def __InitMethods(self):
        for key in self.objs:
            sw = snmp_wapper(self.dev, self.objs[key])
            setattr(self, key, sw.snmpwalk) 
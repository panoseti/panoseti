#!/usr/bin/python3
from typing import Any

import netsnmp

#os.environ['MIBDIRS']='+./'

wrsSnmpObjs={'sfppn'         : 'WR-SWITCH-MIB::wrsPortStatusSfpPN' , \
             'linkstatus'    : 'WR-SWITCH-MIB::wrsPortStatusLink'  , \
             'pllstatus'     : 'WR-SWITCH-MIB::wrsSoftPLLStatus'         }

class snmp_wapper:
    def __init__(self, dev: str, obj: str) -> None:
        self.dev = dev
        self.obj = obj
    def snmpwalk(self) -> int | list[str]:
        oid = netsnmp.Varbind(self.obj)
        try:
            res = netsnmp.snmpwalk(oid, Version=2, DestHost=self.dev, Community='public')
            if(len(res) == 0):
                return 0
            r: list[str] = []
            for i in range(len(res)):
                r.append(res[i].decode('utf-8'))
            return r
        except Exception:
            return -1

class wrs_snmp:    
    def __init__(self, dev: str = '10.0.1.36', objs: Any = wrsSnmpObjs) -> None:
        self.dev = dev
        self.objs = objs
        self.__InitMethods()
    # create methods base on the dict--wrsSnmpObj 
    def __InitMethods(self) -> None:
        for key in self.objs:
            sw = snmp_wapper(self.dev, self.objs[key])
            setattr(self, key, sw.snmpwalk) 
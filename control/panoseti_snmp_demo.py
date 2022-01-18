#!/usr/bin/python3
from panoseti_snmp import wrs_snmp
from panoseti_snmp import wre_snmp

wrs = wrs_snmp('10.1.1.121')
wrs.help()
wrs.wrs_sfp()
wrs.wrs_link()

wre = wre_snmp('192.168.1.99')
wre.help()
wre.wre_sfp()
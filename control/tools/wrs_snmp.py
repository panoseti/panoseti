#!/usr/bin/env python3
from panoseti_snmp import wrs_snmp

#------------------------------------------------------------#

LINK_DOWN	        =	'1'
LINK_UP 	        =	'2'

SFP_PN0 	        = 	'PS-FB-TX1310'
SFP_PN1 	        =	'PS-FB-RX1310'

SOFTPLL_LOCKED      =   '1'
SOFTPLL_UNLOCKED    =   '2'

#------------------------------------------------------------#
# check the PN of SFP transceivers
#
def wrsSFPCheck(wrs):
    res = wrs.sfppn()
    if(res == -1):
        print('************************************************')
        print("We can't connect to WR-SWITCH(%s)!"%(wrs.dev))
        print('************************************************')
    else:
        print('*****************WR-SWITCH SFP CHECK***********************')
        if(res == 0):
            print('WR-SWITCH(%s) : No sfp transceivers detected!' %(wrs.dev))
        else:
            failed = 0
            for i in range(len(res)):
                if(len(res[i]) != 0):
                    if(res[i] != SFP_PN1):
                        failed = 1
                        print('WR-SWITCH(%s) : sfp%2d is %-16s[ FAIL ]' %(wrs.dev, i+1, res[i]))
                    else:
                        print('WR-SWITCH(%s) : sfp%2d is %-16s[ PASS ]' %(wrs.dev, i+1, res[i]))
            if failed == 0:
                print(' ')
                print('WR-SWITCH(%s) : sfp transceivers are checked!' % (wrs.dev))
                print(' ')
            else:
                print(' ')
                print('Error : Please check the sfp transceivers!!')
                print('The part number of the sfp transceiver should be %s'%(SFP_PN1))
                print(' ')

# check the link status
#
def wrsLinkStatusCheck(wrs):
    res = wrs.linkstatus()
    if(res == -1):
        print('********************Error***************************')
        print("We can't connect to WR-Endpoint(%s)!"%(wrs.dev))
        print('****************************************************')
    else:
        print('*****************WR-SWITCH LINK CHECK***********************')
        if(res == 0):
            print('WR-SWITCH(%s) : No sfp transceivers detected!' %(wrs.dev))
        else:
            for i in range(len(res)):
                if res[i] == LINK_UP :
                    print('WR-SWITCH(%s) : Port%2d LINK_UP  ' %(wrs.dev, i+1))
                else:
                    print('WR-SWITCH(%s) : Port%2d LINK_DOWN' %(wrs.dev, i+1))

# check the softpll status
#
def wrsSoftPLLCheck(wrs):
    res = wrs.pllstatus()
    if(res[0] == -1):
        print('********************Error***************************')
        print("We can't connect to WR-Endpoint(%s)!"%(wrs.dev))
        print('****************************************************')
    else:
        print('***************WR-SWITCH SoftPLL CHECK**********************')
        if(res == SOFTPLL_LOCKED):
            print('WR-SWITCH(%s) SoftPLL Status: %s'%(wrs.dev, 'LOCKED'))
        elif(res == SOFTPLL_UNLOCKED):
            print('WR-SWITCH(%s) SoftPLL Status: %s'%(wrs.dev, 'UNLOCK'))
            print('Please Check 10MHz and 1PPS!!!')
        else:
            print('WR-SWITCH(%s) SoftPLL Status: %s(%s)'%(wrs.dev, 'WEIRD STATUS', res[0]))
            print('Please Check 10MHz and 1PPS!!!')
       
def main():
    dev = '10.0.1.36'
    wrs = wrs_snmp(dev)
    wrsSFPCheck(wrs)
    wrsLinkStatusCheck(wrs)
    wrsSoftPLLCheck(wrs)


if __name__ == '__main__':
    main()


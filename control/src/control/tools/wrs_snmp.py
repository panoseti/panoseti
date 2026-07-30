#!/usr/bin/env python3



from control.utils.panoseti_snmp import wrs_snmp

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
def wrsSFPCheck(wrs: wrs_snmp) -> None:
    res = wrs.sfppn() # type: ignore
    if(res == -1):
        print('************************************************')
        print(f"We can't connect to WR-SWITCH({wrs.dev})!")
        print('************************************************')
    else:
        print('*****************WR-SWITCH SFP CHECK***********************')
        if(res == 0):
            print(f'WR-SWITCH({wrs.dev}) : No sfp transceivers detected!')
        elif isinstance(res, list):
            failed = 0
            for i in range(len(res)):
                if(len(res[i]) != 0):
                    if(res[i] != SFP_PN1):
                        failed = 1
                        print(f'WR-SWITCH({wrs.dev}) : sfp{i+1:2d} is {res[i]:-16s}[ FAIL ]')
                    else:
                        print(f'WR-SWITCH({wrs.dev}) : sfp{i+1:2d} is {res[i]:-16s}[ PASS ]')
            if failed == 0:
                print(' ')
                print(f'WR-SWITCH({wrs.dev}) : sfp transceivers are checked!')
                print(' ')
            else:
                print(' ')
                print('Error : Please check the sfp transceivers!!')
                print(f'The part number of the sfp transceiver should be {SFP_PN1}')
                print(' ')

# check the link status
#
def wrsLinkStatusCheck(wrs: wrs_snmp) -> None:
    res = wrs.linkstatus() # type: ignore
    if(res == -1):
        print('********************Error***************************')
        print(f"We can't connect to WR-Endpoint({wrs.dev})!")
        print('****************************************************')
    else:
        print('*****************WR-SWITCH LINK CHECK***********************')
        if(res == 0):
            print(f'WR-SWITCH({wrs.dev}) : No sfp transceivers detected!')
        elif isinstance(res, list):
            for i in range(len(res)):
                if res[i] == LINK_UP :
                    print(f'WR-SWITCH({wrs.dev}) : Port{i+1:2d} LINK_UP  ')
                else:
                    print(f'WR-SWITCH({wrs.dev}) : Port{i+1:2d} LINK_DOWN')

# check the softpll status
#
def wrsSoftPLLCheck(wrs: wrs_snmp) -> None:
    res = wrs.pllstatus() # type: ignore
    if isinstance(res, int) and res == -1:
        print('********************Error***************************')
        print(f"We can't connect to WR-Endpoint({wrs.dev})!")
        print('****************************************************')
    else:
        print('***************WR-SWITCH SoftPLL CHECK**********************')
        if(res == SOFTPLL_LOCKED):
            print('WR-SWITCH({}) SoftPLL Status: {}'.format(wrs.dev, 'LOCKED'))
        elif(res == SOFTPLL_UNLOCKED):
            print('WR-SWITCH({}) SoftPLL Status: {}'.format(wrs.dev, 'UNLOCK'))
            print('Please Check 10MHz and 1PPS!!!')
        elif isinstance(res, list) and len(res) > 0:
            print('WR-SWITCH({}) SoftPLL Status: {}({})'.format(wrs.dev, 'WEIRD STATUS', res[0]))
            print('Please Check 10MHz and 1PPS!!!')
       
def main() -> None:
    dev = '10.0.1.36'
    wrs = wrs_snmp(dev)
    wrsSFPCheck(wrs)
    wrsLinkStatusCheck(wrs)
    wrsSoftPLLCheck(wrs)


if __name__ == '__main__':
    main()


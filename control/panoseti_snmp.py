#!/usr/bin/python3

import os
import netsnmp

LINK_DOWN	=	'1'
LINK_UP 	=	'2'
SFP_PN0 	= 	'PS-FB-TX1310'
SFP_PN1 	=	'PS-FB-RX1310'

os.environ['MIBDIRS']='+./'

class wrs_snmp(object):
	def __init__(self,switch_name='10.1.1.121'):
		self.switch_name = switch_name
	#print help information
	def help(self):
		print('Help Information:')
		print('wrs_sfp      : get the sfp transceivers information on wr-switch')
		print('wrs_link     : get the link status of each port on wr-switch')
	#check the sfp transceivers on WR Switch
	def wrs_sfp(self):
		check_flag = 0
		oid = netsnmp.Varbind('WR-SWITCH-MIB::wrsPortStatusSfpPN')
		try:
			res = netsnmp.snmpwalk(oid, Version=2, DestHost=self.switch_name,Community='public')
		except:
			print('************************************************')
			print("We can't connect to WR-SWITCH(%s)!"%(self.switch_name))
			print('************************************************')
			return
	
		print('*****************WR-SWITCH SFP CHECK***********************')
	
		if(len(res)==0):
			print('WR-SWITCH(%s) : No sfp transceivers detected!' %(self.switch_name))
			return
	
		for i in range(len(res)):
			if len(res[i]) != 0:
				sfp_tmp = bytes.decode(res[i]).replace(' ','') 					#convert bytes to str, and replace the 'space' at the end
				if sfp_tmp != SFP_PN0 and sfp_tmp != SFP_PN1 :
					check_flag = 1
					print('WR-SWITCH(%s) : sfp%2d is %-16s[ FAIL ]' %(self.switch_name, i+1, sfp_tmp))
				else:
					print('WR-SWITCH(%s) : sfp%2d is %-16s[ PASS ]' %(self.switch_name, i+1, sfp_tmp))
		if check_flag == 0:
			print(' ')
			print('WR-SWITCH(%s) : sfp transceivers are checked!' % (self.switch_name))
			print(' ')
		else:
			print(' ')
			print('Error : Please check the sfp transceivers!!')
			print(' ')
		
	#check the link status on WR Switch
	def wrs_link(self):
		oid = netsnmp.Varbind('WR-SWITCH-MIB::wrsPortStatusLink')
		try:
			res = netsnmp.snmpwalk(oid, Version=2, DestHost=self.switch_name, Community='public')
		except:
			print('********************Error***************************')
			print("We can't connect to WR-Endpoint(%s)!"%(self.switch_name))
			print('****************************************************')
			return
	
		print('*****************WR-SWITCH LINK CHECK***********************')
		if(len(res)==0):
			print('WR-SWITCH(%s) : No sfp transceivers detected!' %(self.switch_name))
			return
	
		for i in range(len(res)):
			tmp = bytes.decode(res[i]).replace(' ','') 					#convert bytes to str, and replace the 'space' at the end
			if tmp == LINK_UP :
				print('WR-SWITCH(%s) : Port%2d LINK_UP  ' %(self.switch_name, i+1))
			else:
				print('WR-SWITCH(%s) : Port%2d LINK_DOWN' %(self.switch_name, i+1))
	
		print(' ')

class wre_snmp(object):
	def __init__(self,endpoint_name='10.1.1.121'):
		self.endpoint_name = endpoint_name
	#print out help information
	def help(self):
		print('Help Information:')
		print('wre_sfp      : get the sfp transceivers information on wr-endpoint')			  
	#check the sfp transceivers on WR Endpoint 
	def wre_sfp(self):
		check_flag = 0
		oid = netsnmp.Varbind('WR-WRPC-MIB::wrpcPortSfpPn')
		try:
			res = netsnmp.snmpwalk(oid, Version=2, DestHost=self.endpoint_name, Community='public')
		except:
			print('********************Error***************************')
			print("We can't connect to WR-Endpoint(%s)!"%(self.endpoint_name))
			print('****************************************************')
			return
	
		print('******************WR-ENDPOINT SFP CHECK*********************')
	
		if(len(res)==0):
			print('WR-Endpoint(%s) : No sfp transceivers detected!' %(self.endpoint_name))
			return
	
		sfp_tmp = bytes.decode(res[0]).replace(' ','')
		if sfp_tmp != SFP_PN0 and sfp_tmp != SFP_PN1 :
			check_flag = 1
			print('WR-Endpoint(%s) : sfp transceiver is %-16s[ FAIL ]' %(self.endpoint_name, sfp_tmp))
		else:
			print('WR-Endpoint(%s) : sfp transceiver is %-16s[ PASS ]' %(self.endpoint_name, sfp_tmp))
	
		if check_flag == 0:
			print(' ')
			print('WR-Endpoint(%s) : sfp transceivers are checked!' % (self.endpoint_name))
			print(' ')
		else:
			print(' ')
			print('Error : Please check the sfp transceivers!!')
			print(' ')


		
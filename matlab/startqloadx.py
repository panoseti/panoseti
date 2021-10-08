
import sys
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var1)
try:
	firmwareName = sys.argv[2]
except:
	firmwareName = 'quabo_0116C_23CBEAFB.bin'


print ("Start Py")
from panoseti_tftp import tftpw


client=tftpw(sys.argv[1])
client.put_bin_file(firmwareName)
#client.put_bin_file('quabo_0115_23BD5F46.bin')
#client.put_bin_file('quabo_GOLD_23BD5DA4.bin',0x0)
#client.put_bin_file('quabo_0113_235E7E56.bin')
#client.put_bin_file('quabo_0112_235151CA.bin')
#client.put_bin_file('quabo_0105F_23175810.bin')
#client.put_bin_file('quabo_0105A_230CFB67.bin')
#client.reboot()
#fhand.close()
#time.sleep(.2)
#sock.close()
#quit()
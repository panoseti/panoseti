
import sys
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints var1

print ("Start Py")
from panoseti_tftp import tftpw


client=tftpw(sys.argv[1])
#client.put_bin_file('quabo_0113_235E7E56.bin')
#client.put_bin_file('quabo_0112_235151CA.bin')
#client.put_bin_file('quabo_0105F_23175810.bin')
#client.put_bin_file('quabo_0105E_2315A45E.bin')
#client.put_bin_file('quabo_0105D_23146769.bin')
#client.put_bin_file('quabo_0105C_23133541.bin')
#client.put_bin_file('quabo_0105A_230CFB67.bin')
client.reboot()
#fhand.close()
#time.sleep(.2)
#sock.close()
#quit()
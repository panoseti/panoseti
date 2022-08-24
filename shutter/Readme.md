Programming the Panoseti Shutter Controller

24Aug2022

Compiling and loading code into the shutter controller is done via Arduino IDE version 1.8.19 or newer. 
Version 2 can also be used. (Both work on PC, OSX, and Linux)

File structured:
Top-level folder, 
      All of the files from panoseti/shutter/software/ needs to be in one folder
      called; “PANOSETI-production2”. (Same name as .ino file)

	Note: Board needs to be powered via an adapter board or from the system board.

1)	Open the Arduino IDE, Select; File -> Open, select the “PANOSETI-production2.ino” file
2)	Select, Arduino -> Preference, Set; Sketchbook Location to the top level folder; “PANOSETI-production2”
3)	Select the board to be; “Adafruit Feather M0”
4)	Connect the USB cable from the PC to board
5)	Select correct serial port, Sketch -> Tool -> Port
6)	Select Verify/Compile, if all is well, download the code to the board with Sketch -> download. (check mark and arrow are short cuts)
7)	Done….


Programming the bootloader.

You need to have an ATMEL-ICE or equivalent and download a copy of ATMEL Studio 7 for the PC.  This is only done once when the board is new.

This is the current method we have used to program the bootloader (and or code) into the SAMD21G processor in the shutter controller.

Connect the .05in, 10pin cable to ATMEL-ICE and to connector J23 on the Shutter Controller.
 
The shutter controller board needs external power powered via an adapter board or from the system board.

1)	Start ATMEL Studio 7, on a PC (no Mac or Linux option!)
2)	See section: Flashing a SAMD21 M0 Board with Atmel Studio in Adafruit reference 4 below as a guide.
3)	Select the programming tool, Select Tool->ATMEL-ICE, and set the interface to SWD.
4)	Select the device, Select -> ATSAMD21G18A
5)	Un-set Bootloader Protection Fuse, and set it to 0x07 hex, or zero bytes.
6)	Click Program, and wait for a confirmation that the fuses have been set. Then, click Verify.
7)	Select Erase Flash before programming and Verify flash after programming, Then, click Program. This will erase flash.
8)	Program Binary File, On the sidebar, click Memories, Select the bootloader to load. (It’s on Panoseti GitHub)
9)	Select Erase Flash before programming, and Verify flash after programming. Then, click Program.
10)	After flashing, you'll need to set the BOOTPROT fuse back to an 8kB bootloader size. From Fuses, set BOOTPROT to 0x02 or 8KB and click Program
11)	Done, now you can load code via USB and the Arduino IDE.

~~~
1)	https://www.microchip.com/en-us/development-tool/ATATMEL-ICE
2)	https://www.mouser.com/ProductDetail/Microchip-Technology-Atmel/ATATMEL-ICE-BASIC?qs=KLFHFgXTQiAG498QgmqIdw%3D%3D
3)	https://www.microchip.com/en-us/tools-resources/develop/microchip-studio#Downloads
4)	https://learn.adafruit.com/how-to-program-samd-bootloaders/programming-the-bootloader-with-atmel-studio
5)	Bootloader code: 
        a.	https://github.com/adafruit/uf2-samdx1/releases/tag/v3.14.0
        b.	bootloader-feather_m0-v3.14.0.bin
~~~

One can also use a SEGGER JLink.

~~~
https://www.adafruit.com/product/1369
https://www.adafruit.com/product/3571
~~~

The alternate method is to use a Microchip PICkit- 4, with an AC102015 adapter and Microchip Mplab-x-IDE ver 6.  Works with PC, OSX, and Linux.

I do not have the tool to try this method.

~~~
https://www.microchip.com/en-us/tools-resources/develop/mplab-x-ide
https://www.microchip.com/en-us/development-tool/PG164140
~~~

Adapter Board:

Need to supply +3.3vdc to P20 Pin 3, and ground to P20, pin 4.

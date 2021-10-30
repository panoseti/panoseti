# Modularity and start the board and do the each of the tasks separately based on the action

########Initq######## (Start up the boards)

# VARIABLES
# fpgadir='/home/panosetigraph/panoseti/FPGA/quabo_v0105';

# action 0 rebooting the quabo
# action 1 load silver firmware
# action 2 load gold firmware

quabolist = ['192.168.0.4', '192.168.0.5', '192.168.0.6', '192.168.0.7']

# For all quabos:
# run python script within fpgadir 
# startqx.py (reboot) startqloadx.py (silver) startqloadgoldx.py (gold)
    


######startmodule####### (Start the modules and runs after initq)

# VARIABLES
# acqmode = 0 (House Keeping) 1 (PH Mode), 2 (Image Mode 16 bit) 3 (Dual Mode) 
# 6 (Image Mode 8 bit 7 (Dual Mode 8bit)

# For all quabos:
# startqNph (Start the high voltage)
# Changepeq (Change ph threshold)
# pauseboard (pauses the board)
# stopHVg (Stop the high voltage)


######startqNph#######

# Read "MarocMap.mat" 
# corresponding between quabo pixel number and the physical 2D detector mapping output
# contains marocmap (64x4x2) => (pixel number, quadrant)
#          marocmap16 (16x16x2) => (pixel number, quadrant)

# change the gain 
# change the threshold of the quabos to the high threshold and then 


######changepeq#######


######pauseboard#######


######stopHVg#######

import threading
import subprocess
import time
import signal
import tkinter

class PanosetiControl:
    command_list = []

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

    def add_commands(self, command_arr:list, sync:bool = False) -> int:
        if sync:
            for i in command_arr:
                self.command_list.append(i)
        else:
            self.command_list.append(command_arr)

    def start(self) -> int:
        pass


#command = Command(add, 2, 3)
#print(command.run())

#shell_command = Command("ls", "-l", is_shell_command=True)
#print(shell_command.run())

#shell_command = Command("hashpipe", "-p", "HSD_hashpipe", "-I", "0", "-o",
# "BINDHOST=\"0.0.0.0\"", "-o", "MAXFILESIZE=500", "-o", "SAVELOC=\"./\"", "-o",
#  "CONFIG=\"./module.config\"", "HSD_net_thread", "HSD_compute_thread",  "HSD_output_thread",
#  is_shell_command=True)
#print(shell_command.run())
#while True:
#    time.sleep(1)
#    print(shell_command.output.stdout)

#process = subprocess.Popen(["hashpipe", "-p", "HSD_hashpipe", "-I", "0", "-o",
#"BINDHOST=\"0.0.0.0\"", "-o", "MAXFILESIZE=500", "-o", "SAVELOC=\"./\"", "-o",
#"CONFIG=\"./module.config\"", "HSD_net_thread", "HSD_compute_thread",  "HSD_output_thread"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def test():
    process = subprocess.Popen(["./HSD_init.sh"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def stream_process(process):
        go = process.poll() is None
        for line in process.stdout:
            print(line)
        return go
    
    while True:
        for i in process.stdout:
            print(i)
        time.sleep(0.1)

    #while True:
    #    print(process.stdout)
    #    time.sleep(1)
    #    if sig:
    #        process.send_signal(signal.SIGINT)


def new_test():
    comm = Command("./HSD_init.sh", asyc=False)

    comm.run()

    while True:
        stdout, stderr = comm.get_stdout_update()
        print(stdout)
        time.sleep(1)
        


class Command:
    def __init__(self, command_str : str, asyc = False) -> None:
        self.command_str = command_str
        self._async = asyc

    def run(self) -> int:
        self.process = subprocess.Popen([self.command_str], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
    def get_stdout_update(self):
        return self.process.communicate()


class PanosetiStartUp:
    command_list = []
    def __init__(self) -> None:
        pass

    def add_command(self, command : str):
        self.command_list.append(command)

    def remove_command(self, index : int):
        if index > 0 and index < len(self.command_list):
            del self.command_list[index]
            return 1
        return 0

    def add_commands(self, commands : list):
        for c in commands:
            self.command_list.append(c)

    def remove_commands(self, indicies : list):
        for i in sorted(indicies, reverse=True):
            del self.command_list[i]

    def run(self):
        pass


if __name__ == "__main__":
    test()
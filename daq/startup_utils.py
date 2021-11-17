from abc import abstractmethod
from tkinter import StringVar
from threading import Thread
import subprocess
import time
import signal
from typing import overload


class Process:
    def __init__(self) -> None:
        self.process = None
        self.str_val : StringVar = None
        self.update_thread = Thread(target=self._threaded_update)

    @abstractmethod
    def run(self) -> int:
        pass

    def stop(self) -> int:
        self.process.send_signal(signal.SIGINT)
        self.process.wait()
    
    def is_running(self):
        if self.process == None:
            return False
        
        if self.process.poll() == None:
            return True
        
        if self.process.poll() < 0:
            return False
    
    def get_stdout_update(self):
        return self.process.stdout.readline().decode("utf-8")
    
    def update_to_StringVar(self, str_val : StringVar):
        if self.update_thread != None and self.update_thread.is_alive():
            print("Returned")
            return
        self.update_thread = Thread(target=self._threaded_update)
        self.str_val = str_val
        self.update_thread.start()
    
    def update_to_stdout(self):
        if self.update_thread != None and self.update_thread.is_alive():
            return
        self.update_thread = Thread(target=self._threaded_update)
        self.str_val = None
        self.update_thread.start()

    def _threaded_update(self):
        if self.process == None:
            return
        
        while self.process.poll() == None:
            if self.str_val == None:
                print(self.get_stdout_update())
            else:
                self.str_val.set(self.str_val.get() + self.get_stdout_update())
            time.sleep(0.1)

class DAQ(Process):
    
    def run(self) -> int:
        self.process = subprocess.Popen(["hashpipe", "-p", "HSD_hashpipe", "-I", "0", "-o",
        "BINDHOST=\"0.0.0.0\"", "-o", "MAXFILESIZE=500", "-o", "SAVELOC=./", "-o",
        "CONFIG=./module.config", "HSD_net_thread", "HSD_compute_thread",  "HSD_output_thread"], shell=False, stdout=subprocess.PIPE)

class HK(Process):

    def run(self) -> int:
        self.process = subprocess.Popen(["python", "redisScripts/captureHKPackets.py"], shell=False, stdout=subprocess.PIPE)

class GPS(Process):

    def run(self) -> int:
        self.process = subprocess.Popen(["python", "redisScripts/captureGPSPackets.py"], shell=False, stdout=subprocess.PIPE)

class WR(Process):

    def run(self) -> int:
        self.process = subprocess.Popen(["python", "redisScripts/captureWRPackets.py"], shell=False, stdout=subprocess.PIPE)

    
            
    
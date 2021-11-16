import tkinter as tk
from tkinter import StringVar, filedialog
from threading import Thread
import subprocess
import time
import signal

class DAQ:
    def __init__(self) -> None:
        self.process = None
        self.str_val : StringVar = None
        self.update_thread = Thread(target=self._threaded_update)

    def run(self) -> int:
        self.process = subprocess.Popen(["hashpipe", "-p", "HSD_hashpipe", "-I", "0", "-o",
"BINDHOST=\"0.0.0.0\"", "-o", "MAXFILESIZE=500", "-o", "SAVELOC=./", "-o",
"CONFIG=./module.config", "HSD_net_thread", "HSD_compute_thread",  "HSD_output_thread"], shell=False, stdout=subprocess.PIPE)
    
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


class PanosetiMainControl:
    def __init__(self) -> None:
        self.main_window = tk.Tk()
        self.main_window.geometry("1280x720")
        self.main_frame = tk.Frame(self.main_window)
        self.status_frame = tk.Frame(self.main_frame)

        self.daq_frame = tk.Frame(self.status_frame)
        self.daq_process = DAQ()
        self.daq_message = StringVar()
        tk.Message(self.daq_frame, textvariable=self.daq_message).pack()
        self.daq_button_message = StringVar()
        self.daq_button_message.set("Start Data Acqusition")
        self.daq_button = tk.Button(self.daq_frame, textvariable=self.daq_button_message, command=self.daq).pack()
        self.daq_frame.pack(side=tk.LEFT)

        self.redis_frame = tk.Frame(self.status_frame)
        tk.Message(self.redis_frame, text="Redis Message").pack()
        tk.Button(self.redis_frame, text="Redis").pack()
        self.redis_frame.pack(side=tk.RIGHT)
        
        self.status_frame.pack()

        self.script_frame = tk.Frame(self.main_frame)
        tk.Button(self.script_frame, text="Script").pack()
        self.script_frame.pack()

        tk.Button(self.main_frame, text="START").pack()

        self.main_frame.pack()
        self.create_menu_bar()

    def create_menu_bar(self):
        menubar = tk.Menu(self.main_window)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save Config", command=self.save_config)
        filemenu.add_command(label="Load Config", command=self.load_config)
        filemenu.add_command(label="Exit", command=self.exit)
        menubar.add_cascade(label="File", menu=filemenu)

        self.main_window.config(menu=menubar)

    
    def save_config(self):
        filename = filedialog.asksaveasfile(title="Save Config File As",
                                            initialdir="~",
                                            filetypes=(("Configuration Files", "*.config"), ("All Files", "*.*")))
        if not filename:
            return
        print(filename)

    def load_config(self):
        filename = filedialog.askopenfile(title="Load Config File From",
                                        initialdir="~", 
                                        filetypes=(("Configuration Files", "*.config"), ("All Files", "*.*")))
        if not filename:
            return

        print(filename.name)

    def exit(self):
        exit(0)
    
    def daq(self):
        if self.daq_process.is_running() or self.daq_process.update_thread.is_alive():
            self.daq_process.stop()
            if not self.daq_process.is_running():
                self.daq_button_message.set("Start Data Acqusition")
            return
        
        self.daq_message.set("")
        self.daq_process.run()
        self.daq_process.update_to_StringVar(self.daq_message)
        #self.daq_process.update_to_stdout()
        self.daq_button_message.set("Stop Data Acqusition")
    
    def test_running(self):
        print(self.daq_process.is_running())
        



if __name__ == "__main__":
    PanosetiMainControl()
    tk.mainloop()
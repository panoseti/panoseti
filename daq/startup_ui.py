import tkinter as tk
from tkinter import StringVar, filedialog
from tkinter.constants import LEFT
from startup_utils import DAQ, HK, GPS, WR


class PanosetiMainControl:
    def __init__(self) -> None:
        self.main_window = tk.Tk()
        self.main_window.geometry("1280x720")
        self.main_frame = tk.Frame(self.main_window)
        self.status_frame = tk.Frame(self.main_frame)

        self.daq_frame = tk.Frame(self.status_frame)
        self.daq_process = DAQ()
        self.daq_message = StringVar()
        tk.Message(self.daq_frame, textvariable=self.daq_message, justify=LEFT).pack()
        self.daq_button_message = StringVar()
        self.daq_button_message.set("Start Data Acqusition")
        self.daq_button = tk.Button(self.daq_frame, textvariable=self.daq_button_message, command=self.daq).pack()
        self.daq_frame.pack(side=tk.LEFT)

        self.redis_frame = tk.Frame(self.status_frame)
        self.redis_HK_process = HK()
        self.redis_HK_message = StringVar()
        self.redis_GPS_process = GPS()
        self.redis_GPS_message = StringVar()
        self.redis_WR_process = WR()
        self.redis_WR_message = StringVar()
        tk.Message(self.redis_frame, textvariable=self.redis_HK_message).pack()
        tk.Message(self.redis_frame, textvariable=self.redis_GPS_message).pack()
        tk.Message(self.redis_frame, textvariable=self.redis_WR_message).pack()
        self.redis_button_message = StringVar()
        self.redis_button_message.set("Start Redis Scripts")
        tk.Button(self.redis_frame, textvariable=self.redis_button_message, command=self.redis_scripts).pack()
        self.redis_frame.pack(side=tk.RIGHT)
        
        self.status_frame.pack()

        self.script_frame = tk.Frame(self.main_frame)
        tk.Button(self.script_frame, text="Preview").pack()
        self.script_frame.pack()

        self.start_message = StringVar()
        self.start_message.set("Start")
        tk.Button(self.main_frame, textvariable=self.start_message, command=self.start).pack()

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
            return 0
        
        self.daq_message.set("")
        self.daq_process.run()
        self.daq_process.update_to_StringVar(self.daq_message)
        self.daq_button_message.set("Stop Data Acqusition")
        return 1
    
    def redis_scripts(self):
        if self.redis_HK_process.is_running() or self.redis_GPS_process.is_running() or self.redis_WR_process.is_running():
            if self.redis_HK_process.is_running():
                self.redis_HK_process.stop()
            
            if self.redis_GPS_process.is_running():
                self.redis_GPS_process.stop()
            
            if self.redis_WR_process.is_running():
                self.redis_WR_process.stop()
            
            if not self.redis_HK_process.is_running() or not self.redis_GPS_process.is_running() or not self.redis_WR_process.is_running():
                self.redis_button_message.set("Start Redis Scripts")
            return 0
        
        self.redis_HK_message.set("")
        self.redis_GPS_message.set("")
        self.redis_WR_message.set("")
        
        self.redis_HK_process.run()
        self.redis_GPS_process.run()
        self.redis_WR_process.run()

        self.redis_HK_process.update_to_StringVar(self.redis_HK_message)
        self.redis_GPS_process.update_to_StringVar(self.redis_GPS_message)
        self.redis_WR_process.update_to_StringVar(self.redis_WR_message)

        self.redis_button_message.set("Stop Redis Scripts")
        return 1
    
    def start(self):
        if self.daq() and self.redis_scripts():
            self.start_message.set("Stop")
            return 1
        self.start_message.set("Start")
        return 0


if __name__ == "__main__":
    PanosetiMainControl()
    tk.mainloop()
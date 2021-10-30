import tkinter as tk
from tkinter import StringVar, filedialog

class PanosetiMainControl:
    def __init__(self) -> None:
        self.main_window = tk.Tk()
        self.main_frame = tk.Frame(self.main_window)

        self.status_frame = tk.Frame(self.main_frame)

        self.hashpipe_frame = tk.Frame(self.status_frame)
        self.hashpipe_message = StringVar()
        tk.Message(self.hashpipe_frame, textvariable=self.hashpipe_message).pack()
        self.hashpipe_message.set("Hashpipe Message")
        self.hashpipe_message.set(self.hashpipe_message.get() + "Changed")
        tk.Button(self.hashpipe_frame, text="Hashpipe").pack()
        self.hashpipe_frame.pack(side=tk.LEFT)

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


PanosetiMainControl()


tk.mainloop()
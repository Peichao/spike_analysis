#!/usr/bin/env python
"""
phy_manual.py: allows selection of *.kwik file and opens manual clustering GUI
author: @anupam
"""

import tkinter as tk
from tkinter import filedialog
import phy
from phy import gui
from phy.session import Session

tk.Tk().withdraw()
kwik_path = tk.filedialog.askopenfilename()

phy.gui.enable_qt()
session = Session(kwik_path, channel_group=1)
session.show_gui()

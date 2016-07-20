#!/usr/bin/env python

"""kilosort_ori.py: creation of size tuning curves based on results of KiloSort."""

import datetime
import tkinter as tk
from tkinter import filedialog
import kilosort_analysis

# ---------------------------------------------------------------------------------------------------------------------#
# For size tuning, use code below:
tk.Tk().withdraw()
kilo_path = tk.filedialog.askdirectory()

fs = 25000
num_aux = int(input(datetime.datetime.now().strftime("%X") + " Number of aux channels recorded?: "))
spike_info, cgs = kilosort_analysis.get_kilo_info(kilo_path, fs)
dig_aux, exp_timing, stim_time = kilosort_analysis.get_aux_info(kilo_path, num_aux)
exp_timing_spike_counts = kilosort_analysis.get_condition_data(spike_info, exp_timing)
kilosort_analysis.cluster_sz_led_plot(cgs, exp_timing_spike_counts, kilo_path, fs)

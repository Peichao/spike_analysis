#!/usr/bin/env python

"""kilosort_ori.py: creation of orientation tuning curves based on results of KiloSort."""

import tkinter as tk
from tkinter import filedialog
import kilosort_analysis

# ---------------------------------------------------------------------------------------------------------------------#
# For orientation tuning, use code below:
tk.Tk().withdraw()
kilo_path = tk.filedialog.askdirectory()

fs = 25000
# num_aux = int(input(datetime.datetime.now().strftime("%X") + " Number of aux channels recorded?: "))
num_aux = 6
spike_info, cgs = kilosort_analysis.get_kilo_info(kilo_path, fs)
dig_aux, exp_timing, stim_time = kilosort_analysis.get_aux_info(kilo_path, num_aux)
exp_timing_spike_counts = kilosort_analysis.get_condition_data(spike_info, exp_timing)
cgs = kilosort_analysis.cluster_ori_led_plot(cgs, exp_timing_spike_counts, kilo_path, fs)
kilosort_analysis.plot_max_rates(cgs)

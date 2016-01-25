#!/usr/bin/env python
"""
analyzer.py: opens analyzer file and extracts trial_num and stim_num
author: @anupam
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np
import scipy.io as sio


def looper():
    tk.Tk().withdraw()
    analyzer_path = tk.filedialog.askopenfilename()

    analyzer_complete = sio.loadmat(analyzer_path, squeeze_me=True, struct_as_record=False)
    analyzer = analyzer_complete['Analyzer']

    b_flag = 0
    if analyzer.loops.conds[len(analyzer.loops.conds) - 1].symbol[0] == 'blank':
        b_flag = 1

    cutoff = 0
    if b_flag == 1:
        cutoff = 2

    trial_num = np.ones((len(analyzer.loops.conds) * len(analyzer.loops.conds[0].repeats),
                         (len(analyzer.loops.conds[0].symbol))))

    for count in range(0, len(analyzer.loops.conds) - cutoff):
        trial_vals = np.zeros(len(analyzer.loops.conds[count].symbol))

        for count2 in range(0, len(analyzer.loops.conds[count].symbol)):
            trial_vals[count2] = analyzer.loops.conds[count].val[count2]

        for count3 in range(0, len(analyzer.loops.conds[count].repeats)):
            aux_trial = analyzer.loops.conds[count].repeats[count3].trialno
            trial_num[aux_trial - 1, :] = trial_vals

    stim_time = np.zeros(3)
    for count4 in range(0, 3):
        stim_time[count4] = analyzer.P.param[count4][2]

    sum_trial = trial_num[:, 0] + trial_num[:, 1]
    [blank_trial_idx] = np.where(sum_trial == 1)
    trial_num[blank_trial_idx, 0] = 256

    return trial_num, stim_time

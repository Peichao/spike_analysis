#!/usr/bin/env python
"""
analyzer.py: opens analyzer file and extracts trial_num and stim_num
author: @anupam
"""

import os
import numpy as np
import scipy.io as sio


def looper_led(analyzer_path):
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


def looper_led_blanks(analyzer_path):
    analyzer_complete = sio.loadmat(analyzer_path, squeeze_me=True, struct_as_record=False)
    analyzer = analyzer_complete['Analyzer']

    b_flag = 0
    if analyzer.loops.conds[len(analyzer.loops.conds) - 1].symbol[0] == 'blank':
        b_flag = 2

    if b_flag == 0:
        trial_num = np.zeros((len(analyzer.loops.conds) * len(analyzer.loops.conds[0].repeats),
                             (len(analyzer.loops.conds[0].symbol))))
    else:
        trial_num = np.zeros(((len(analyzer.loops.conds) - b_flag) * len(analyzer.loops.conds[0].repeats) +
                              len(analyzer.loops.conds[-2].repeats) + len(analyzer.loops.conds[-1].repeats),
                              (len(analyzer.loops.conds[0].symbol))))

    for count in range(0, len(analyzer.loops.conds) - b_flag):
        trial_vals = np.zeros(len(analyzer.loops.conds[count].symbol))

        for count2 in range(0, len(analyzer.loops.conds[count].symbol)):
            trial_vals[count2] = analyzer.loops.conds[count].val[count2]

        for count3 in range(0, len(analyzer.loops.conds[count].repeats)):
            aux_trial = analyzer.loops.conds[count].repeats[count3].trialno
            trial_num[aux_trial - 1, :] = trial_vals

    for blank_trial_1 in range(0, len(analyzer.loops.conds[-2].repeats)):
        aux_trial = analyzer.loops.conds[-2].repeats[blank_trial_1].trialno
        light_cond = trial_num[aux_trial - 2, 1]
        trial_num[aux_trial - 1, :] = np.array([256, light_cond])

    for blank_trial_2 in range(0, len(analyzer.loops.conds[-1].repeats)):
        aux_trial = analyzer.loops.conds[-1].repeats[blank_trial_2].trialno
        light_cond = trial_num[aux_trial - 2, 1]
        trial_num[aux_trial - 1, :] = np.array([256, light_cond])

    stim_time = np.zeros(3)
    for count4 in range(0, 3):
        stim_time[count4] = analyzer.P.param[count4][2]

    return trial_num, stim_time


def get_analyzer_path(folder_path):
    analyzer_name = []
    analyzer_name += [file for file in os.listdir(folder_path) if file.endswith('.analyzer')]
    analyzer_path = folder_path + '/' + analyzer_name[0]
    return analyzer_path

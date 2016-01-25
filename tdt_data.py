#!/usr/bin/env python
"""
tdt_data.py: functions to extract TDT data using matlab engine, converts data to numpy arrays for analysis in python
author: @anupam
"""
import os
import tkinter as tk
from tkinter import filedialog
import matlab.engine
import numpy as np


def get_tank_path():
    """
    gets path to TDT tank directory from file dialog
    :return: path to tdt tank folder
    """
    tk.Tk().withdraw()
    tank_path = tk.filedialog.askdirectory()
    return tank_path


def get_block():
    """
    asks for user input for string of block name to be analyzed
    :return: name of block desired to save
    """
    block = input("Enter block name: ")
    return block


def extract_matlab_data(tank_path, block):
    """
    uses matlab engine to extract raw TDT data into numpy array (nsamples x nchannels)
    :param tank_path: string path to tank to be saved as dat file
    :param block: string block of block to be saved
    :return: numpy array of raw data (nsamples x nchannels)
    """
    eng = matlab.engine.start_matlab()
    raws = eng.tdt_data_py(tank_path, block)
    raws_array = np.array(raws._data).reshape(raws.size[::-1]).T
    raws_array = np.transpose(raws_array)
    eng.quit()
    return raws_array


def extract_ori(tank_path, block):
    eng = matlab.engine.start_matlab()
    [lfpsi, time_index, epocs] = eng.tdt_LFP_py(tank_path, block, nargout=3)

    time_index_array = np.array(time_index._data).reshape(time_index.size[::-1]).T
    epocs_array = np.array(epocs._data).reshape(epocs.size[::-1]).T

    eng.quit()
    return lfpsi, time_index_array, epocs_array


def save_matlab_dat(tank_path, block):
    """
    saves raw TDT data as int16 binary dat file
    :param tank_path: string path to tank to be saved as dat file
    :param block: string block of block to be saved
    :return: numpy array of raw data (nsamples x nchannels)
    """
    directory = os.path.abspath(os.path.join(tank_path, os.pardir))
    if not os.path.exists(directory + "/%s" % block):
        os.makedirs(directory + "/%s" % block)
    raw = extract_matlab_data(tank_path, block)
    raw.tofile(directory + '/' + block + '/' + block + '.dat')
    return raw

#!/usr/bin/env python

import os
import phy_analysis
import analyzer
import tdt_data
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt

params = {'backend': 'Agg',
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'font.size': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'savefig.dpi': 600,
          'font.family': 'sans-serif',
          'axes.linewidth': 0.5,
          'xtick.major.size': 2,
          'ytick.major.size': 2,
          }
rcParams.update(params)
plt.ioff()

# get input file information
kwik_path = phy_analysis.get_kwik_path()
tank_path = tdt_data.get_tank_path()
block = tdt_data.get_block()

beginning_directory = os.getcwd()
directory = os.path.abspath(os.path.join(kwik_path, os.pardir))
if not os.path.exists(directory + "/figures"):
    os.makedirs(directory + "/figures")

trial_num, stim_time = analyzer.looper()

channel_group = 0
cluster_ids_dict = phy_analysis.extract_spike_times(kwik_path, channel_group)
model = phy_analysis.get_kwik_model(kwik_path, channel_group)
lfpsi, time_index, epocs = tdt_data.extract_ori(tank_path, block)

# figure out number of samples during pre, post, and stim times
pre = stim_time[0]
pre_samples = np.round(pre/lfpsi)  # number of samples in pre-stim time
post = stim_time[1]
post_samples = np.round(post/lfpsi)  # number of samples in post-stim time
stim_length = stim_time[2]
stim_samples = np.round(stim_length/lfpsi)  # number of samples in stimulus period
total_length_samples = np.round((pre+stim_length+post)/lfpsi)  # total length in samples

# figure out trial start and stop times using Epocs from TDT
start_time = epocs[1, :]
stop_time = start_time + pre + stim_length + post
last_trial = np.shape(epocs)[1]

# create trial matrix with the entire stimulus time (pre, stim, post)
trials = np.zeros((len(start_time), 2))
for iTR in range(0, len(start_time)):
    trials[iTR, 0] = np.where(time_index[0, :] >= start_time[iTR])[0][0] - 1
    trials[iTR, 1] = np.where(time_index[0, :] >= stop_time[iTR])[0][0] - 1

trials_sub = trials[:, 1] - trials[:, 0]  # in case trial timing is off

# correct indexing to make all trials of equal number of samples due to timing being off
wrong = np.where(trials_sub == total_length_samples + 1)
trials[wrong, 0] += 1
wrong = np.where(trials_sub == total_length_samples - 1)
trials[wrong, 0] -= 1

# create a nested dictionary of cluster spike times by trial within dictionary of clusters
cluster_spike_trials = {}
for cluster in cluster_ids_dict:
    trial_dict = {}
    cluster_spike_times = np.array(cluster_ids_dict[cluster])
    for trial in range(0, len(trials)):
        trial_spikes_idx = np.where((cluster_spike_times >= (trials[trial, 0] + pre_samples) * lfpsi) &
                                    (cluster_spike_times <= (trials[trial, 1] - post_samples) * lfpsi))
        trial_dict[trial] = cluster_spike_times[trial_spikes_idx]
    cluster_spike_trials[cluster] = trial_dict

# get LED data from analyzer file
trial_params = np.size(trial_num, 1)
if trial_params == 2:
    LED_on = np.where(trial_num[:, 1] == 1)
    LED_off = np.where(trial_num[:, 1] == 0)
    columns = 2
else:
    LED_on = np.where(trial_num[:, 2] == 1)
    LED_off = np.where(trial_num[:, 2] == 0)
    columns = 3

blank_trials = np.where(trial_num[:, 0] == 1)
vstim_trials = np.where(trial_num[:, 0] != 1)
vstim_LEDon_trials = np.where((trial_num[:, 0] != 1) & (trial_num[:, 1] == 1))
vstim_LEDoff_trials = np.where((trial_num[:, 0] != 1) & (trial_num[:, 1] == 0))

q = np.max(trial_num[:, 0])
if q == 315:
    deg = np.arange(0, 360, 45)
else:
    deg = np.arange(0, 360, 30)

os.chdir(directory + "/figures")

spike_rates = {}
spike_rates_LEDon = {}
spike_rates_LEDoff = {}
for cluster in cluster_spike_trials:
    plt.figure()

    cluster_ori = {}
    cluster_ori_LEDon = {}
    cluster_ori_LEDoff = {}

    cluster_ori_rates = {}
    cluster_ori_rates_LEDon = {}
    cluster_ori_rates_LEDoff = {}

    for ori in deg:
        cluster_ori[ori] = np.array([])
        cluster_ori_LEDon[ori] = np.array([])
        cluster_ori_LEDoff[ori] = np.array([])
        ori_trials = np.where(trial_num[:, 0] == ori)
        ori_trials_LEDon = np.where((trial_num[:, 0] == ori) & (trial_num[:, 1] == 1))
        ori_trials_LEDoff = np.where((trial_num[:, 0] == ori) & (trial_num[:, 1] == 0))

        if np.shape(ori_trials)[1] > 0:
            for trial in np.nditer(ori_trials):
                cluster_ori[ori] = np.r_[cluster_ori[ori], cluster_spike_trials[cluster][trial.tolist()]]
            cluster_ori_rates[ori] = len(cluster_ori[ori])/(np.size(ori_trials) * (pre + post + stim_length))

        if np.shape(ori_trials_LEDon)[1] > 0:
            for trial in np.nditer(ori_trials_LEDon):
                cluster_ori_LEDon[ori] = np.r_[cluster_ori_LEDon[ori], cluster_spike_trials[cluster][trial.tolist()]]
            cluster_ori_rates_LEDon[ori] = len(cluster_ori_LEDon[ori])/(np.size(ori_trials_LEDon) *
                                                                        (pre + post + stim_length))

        if np.shape(ori_trials_LEDoff)[1] > 0:
            for trial in np.nditer(ori_trials_LEDoff):
                cluster_ori_LEDoff[ori] = np.r_[cluster_ori_LEDoff[ori], cluster_spike_trials[cluster][trial.tolist()]]
            cluster_ori_rates_LEDoff[ori] = len(cluster_ori_LEDoff[ori])/(np.size(ori_trials_LEDoff) *
                                                                          (pre + post + stim_length))

    spike_rates[cluster] = cluster_ori_rates
    spike_rates_LEDon[cluster] = cluster_ori_rates_LEDon
    spike_rates_LEDoff[cluster] = cluster_ori_rates_LEDoff

    spike_rates_pd = pd.DataFrame([spike_rates[cluster]], index=['all']).transpose()
    spike_rates_LEDon_pd = pd.DataFrame([spike_rates_LEDon[cluster]], index=['LED on']).transpose()
    spike_rates_LEDoff_pd = pd.DataFrame([spike_rates_LEDoff[cluster]], index=['LED off']).transpose()

    all_rates = pd.concat([spike_rates_LEDon_pd, spike_rates_LEDoff_pd], axis=1, join_axes=[spike_rates_pd.index])
    ax = all_rates.plot(linewidth=3)
    fig = ax.get_figure()

    plt.ylim(ymin=0)
    plt.xlim(0, 315)
    plt.xlabel('Orientation (degrees)')
    plt.ylabel('Spike Rate (spikes/sec)')
    plt.suptitle('Orientation Selectivity, Cluster %d' % cluster, fontsize=14, fontweight='bold')
    plt.title('Manually Sorted, %s' % model.default_cluster_groups[model.cluster_groups[cluster]])
    plt.savefig("cluster%d.png" % cluster)
    plt.close(fig)

os.chdir(beginning_directory)

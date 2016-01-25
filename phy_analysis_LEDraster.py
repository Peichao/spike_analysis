#!/usr/bin/env python
"""
spikesort.py: creates a raster plot of a cluster showing spikes before/during/after each trial
computes Kruskal-Wallis statistical analysis and adds H-statistic and p-value to the plot
author: @anupam
"""

import tkinter as tk
from tkinter import filedialog
import os
from phy.io import KwikModel
import numpy as np
import scipy.io as sio
from scipy.stats import mstats
import matplotlib.pyplot as plt

LED_freq = 20  # Hz
num_trials = 20
trial_length = 2  # seconds

# receive user input of .kwik and .mat files, LED frequency, and number of trials
tk.Tk().withdraw()
kwik_path = tk.filedialog.askopenfilename()
mat_path = tk.filedialog.askopenfilename()

directory = os.path.abspath(os.path.join(kwik_path, os.pardir))
if not os.path.exists(directory + "/figures"):
    os.makedirs(directory + "/figures")
os.chdir(directory + "/figures")

# load .mat file and calculate trial start times, assuming trials of length trial_length
mat = sio.loadmat(mat_path)
LeON_timestamps = mat['LeON_timestamps']
trial_start_times = np.zeros(num_trials)
for trial in range(0, num_trials):
    trial_start_times[trial] = LeON_timestamps[0, trial*LED_freq * trial_length]

# load .kwik file and create dictionary with spike times of each cluster
model = KwikModel(kwik_path)
model.clustering = 'main'
model.describe()

spike_times = model.spike_times
spike_clusters = model.spike_clusters
cluster_ids = model.cluster_ids

cluster_ids_dict = {key: [] for key in cluster_ids}
for spike_time_idx in range(0, len(spike_times)):
    cluster_ids_dict[spike_clusters[spike_time_idx]].append(spike_times[spike_time_idx])

# for each cluster, create dictionary of trials with spike times
# create raster plot of trials for each cluster and save in new folder named figures
cluster_trials_dict = {trial: [] for trial in range(0, len(trial_start_times))}
spike_counts = np.zeros((len(trial_start_times), 3))

for cluster in cluster_ids_dict:
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.55, 0.75])

    for trial in range(0, len(trial_start_times)):
        cluster_spike_times = np.array(cluster_ids_dict[cluster])
        start_time = trial_start_times[trial]
        end_time = start_time + trial_length
        spike_times_idx = np.where((cluster_spike_times >= start_time - 1) & (cluster_spike_times <= end_time + 1))
        cluster_trials_dict[trial] = cluster_spike_times[spike_times_idx] - start_time
        ax.vlines(cluster_trials_dict[trial], trial + 0.5, trial + 1.5, color='k')

        # create array of spike counts before, during, and after each trial
        spike_counts[trial, 0] = np.size(np.where((cluster_spike_times >= start_time) &
                                                  (cluster_spike_times <= end_time)))
        spike_counts[trial, 1] = np.size(np.where((cluster_spike_times >= start_time - trial_length) &
                                                  (cluster_spike_times <= start_time)))
        spike_counts[trial, 2] = np.size(np.where((cluster_spike_times >= end_time) &
                                                  (cluster_spike_times <= end_time + trial_length)))

    spike_rate_averaged = np.average(spike_counts, axis=0) / 2

    # computer Kruskal-Wallis statistical analysis on cluster data
    H, pval = mstats.kruskalwallis(spike_counts[:, 0], spike_counts[:, 1], spike_counts[:, 2])

    np.savetxt("cluster%d.csv" % cluster, spike_counts, delimiter=",")
    ax.axvspan(0, trial_length, facecolor='b', alpha=0.5)
    fig.suptitle("Cluster %d, '%s'" % (cluster, model.default_cluster_groups[model.cluster_groups[cluster]]),
                 fontsize=14, fontweight='bold')
    ax.set_title('H-statistic = %f, p-value = %f' % (H, pval))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trial Number")
    ax.axis([-1, trial_length + 1, 0.5, len(cluster_trials_dict) + 0.5])
    plt.text(trial_length + 1.5, num_trials / 2, "Average Spike Rates\nBefore = %f\nDuring = %f\nAfter = %f" %
             (spike_rate_averaged[0], spike_rate_averaged[1], spike_rate_averaged[2]),
             bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
    plt.savefig("cluster%d.png" % cluster)
    plt.close()



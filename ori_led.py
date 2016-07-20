#!/usr/bin/env python
"""
ori_led.py: takes sorted data and creates orientation tuning curves, separating for LED on/off states
author: @anupam
"""
import os
import pickle
from pathlib import Path
import phy_analysis
import analyzer
import tdt_data
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt


def ori_plot(cluster, LEDon_rates, model, LEDoff_rates, collapsed_means_on,
             collapsed_means_off, w_p_sig, P, primary_chan_dict):
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

    mean_both = pd.concat([LEDon_rates.mean(), LEDoff_rates.mean()], axis=1)
    mean_both.columns = ['LED on', 'LED off']

    mean_both_collapsed = pd.concat([collapsed_means_on.mean(), collapsed_means_off.mean()], axis=1)
    mean_both_collapsed.columns = ['LED on', 'LED off']

    ax1 = plt.subplot(211)
    mean_both.plot(linewidth=2, ax=ax1)
    ax1.fill_between(np.array(LEDon_rates.sem().keys()), LEDon_rates.mean().values - LEDon_rates.sem().values,
                     LEDon_rates.mean().values + LEDon_rates.sem().values,
                     alpha=0.3)
    ax1.fill_between(np.array(LEDoff_rates.sem().keys()), LEDoff_rates.mean().values - LEDoff_rates.sem().values,
                     LEDoff_rates.mean().values + LEDoff_rates.sem().values,
                     alpha=0.3, edgecolor='#3F7F4C', facecolor='#7EFF99')

    ax2 = plt.subplot(212)
    mean_both_collapsed.plot(linewidth=2, ax=ax2)
    ax2.fill_between(np.array(collapsed_means_on.sem().keys()),
                     collapsed_means_on.mean().values - collapsed_means_on.sem().values,
                     collapsed_means_on.mean().values + collapsed_means_on.sem().values,
                     alpha=0.3)
    ax2.fill_between(np.array(collapsed_means_off.sem().keys()),
                     collapsed_means_off.mean().values - collapsed_means_off.sem().values,
                     collapsed_means_off.mean().values + collapsed_means_off.sem().values,
                     alpha=0.3, edgecolor='#3F7F4C', facecolor='#7EFF99')

    for sig in w_p_sig[cluster]:
        ax2.axvline(sig, linewidth=3, color='r')

    plt.suptitle('Orientation Selectivity, Cluster %d' % cluster, fontsize=14, fontweight='bold')
    ax1.set_title('Manually Sorted, %s, p = %f, Primary Channel = %d' %
                  (model.cluster_groups[cluster],
                   P[cluster], primary_chan_dict[cluster]))
    plt.subplots_adjust(hspace=.3)

    ax1.set_ylim(ymin=0)
    ax1.set_xlim(0, 315)
    ax1.set_xlabel('Orientation (degrees)')
    ax1.set_ylabel('Spike Rate (spikes/sec)')
    ax2.set_xlim(0, 135)
    ax2.set_xlabel('Orientation (degrees)')
    ax2.set_ylabel('Baseline-Subtracted Spike Rate (spikes/sec)')

    os.chdir(model.cluster_groups[cluster])
    plt.savefig("cluster%d.png" % cluster)
    plt.savefig("cluster%d.eps" % cluster, format='eps')
    os.chdir(str(Path(os.getcwd()).parent))

    plt.close()


def ori_led(date, block, channel_group, analyzer_path):
    beginning_directory = os.getcwd()

    # get input file information
    trial_num, stim_time = analyzer.looper_led(analyzer_path)

    kwik_path = "C:/Users/anupam/Desktop/salk data/VIPCR/%s/%s/%s.kwik" % (date, block, block)
    tank_path = "C:/Users/anupam/Desktop/salk data/VIPCR/%s/%s" % (date, date)

    cluster_ids_dict = phy_analysis.extract_spike_times(kwik_path, channel_group)
    model = phy_analysis.get_kwik_model(kwik_path, channel_group)
    lfpsi, time_index, epocs = tdt_data.extract_ori(tank_path, block)

    # find primary channel of each cluster and save to dictionary
    primary_chan_dict = phy_analysis.primary_channel(kwik_path, block, channel_group)

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
    last_trial = np.size(epocs)

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
    cluster_spike_trials_pre = {}
    cluster_spike_rate_pre = {}
    for cluster in cluster_ids_dict:
        all_pre = np.array([])
        trial_dict = {}
        pre_trial_dict = {}
        cluster_spike_times = np.array(cluster_ids_dict[cluster])
        for trial in range(0, len(trials)):
            pre_trial_idx = np.where((cluster_spike_times >= trials[trial, 0] * lfpsi) &
                                     (cluster_spike_times <= (trials[trial, 0] + pre_samples) * lfpsi))
            trial_spikes_idx = np.where((cluster_spike_times >= (trials[trial, 0] + pre_samples) * lfpsi) &
                                        (cluster_spike_times <= (trials[trial, 1] - post_samples) * lfpsi))
            pre_trial_dict[trial] = cluster_spike_times[pre_trial_idx]
            trial_dict[trial] = cluster_spike_times[trial_spikes_idx]
            all_pre = np.append(all_pre, pre_trial_dict[trial])

        cluster_spike_trials[cluster] = trial_dict
        cluster_spike_trials_pre[cluster] = pre_trial_dict
        pre_trial_total = pre * len(trials)
        cluster_spike_rate_pre[cluster] = np.size(all_pre) / pre_trial_total

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

    # get lists of blank trials, trials with visual stimulus, and visual stimulus trials with LED on/LED off
    blank_trials = np.where(trial_num[:, 0] == 1)
    vstim_trials = np.where(trial_num[:, 0] != 1)
    vstim_LEDon_trials = np.where((trial_num[:, 0] != 1) & (trial_num[:, 1] == 1))
    vstim_LEDoff_trials = np.where((trial_num[:, 0] != 1) & (trial_num[:, 1] == 0))

    # create numpy array of orientations used in experiment
    q = np.max(trial_num[:, 0])
    if q == 315:
        deg = np.arange(0, 360, 45)
    else:
        deg = np.arange(0, 360, 30)

    # change to directory to save figures
    directory = os.path.abspath(os.path.join(kwik_path, os.pardir))
    save_dir = ("%s/figures/channel_group_%d" % (directory, channel_group))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    # create directories for each type of cluster (good, MUA, etc.)
    for clus in model.cluster_groups:
        if not os.path.exists(save_dir + '/' + model.cluster_groups[clus]):
            os.makedirs(model.cluster_groups[clus])

    P = {}
    w_p_p = {}
    w_p_sig = {}
    max_rate_on = {}
    max_rate_off = {}

    for cluster in cluster_spike_trials:

        LEDon_rates = pd.DataFrame()
        LEDoff_rates = pd.DataFrame()
        collapsed_means_on = pd.DataFrame()
        collapsed_means_off = pd.DataFrame()

        for ori in deg:

            ori_trials = np.where(trial_num[:, 0] == ori)
            ori_trials_LEDon = np.where((trial_num[:, 0] == ori) & (trial_num[:, 1] == 1))
            ori_trials_LEDoff = np.where((trial_num[:, 0] == ori) & (trial_num[:, 1] == 0))

            if np.size(ori_trials_LEDon) > 0:
                cluster_ori_LEDon_rates = pd.Series()
                for trial in np.nditer(ori_trials_LEDon):
                    cluster_ori_LEDon_rate = np.size(cluster_spike_trials[cluster][trial.tolist()]) / stim_length
                    cluster_ori_LEDon_rates = cluster_ori_LEDon_rates.append(pd.Series([cluster_ori_LEDon_rate]))
                LEDon_rates[ori] = cluster_ori_LEDon_rates

            if np.size(ori_trials_LEDoff) > 0:
                cluster_ori_LEDoff_rates = pd.Series()
                for trial in np.nditer(ori_trials_LEDoff):
                    cluster_ori_LEDoff_rate = np.size(cluster_spike_trials[cluster][trial.tolist()]) / stim_length
                    cluster_ori_LEDoff_rates = cluster_ori_LEDoff_rates.append(pd.Series([cluster_ori_LEDoff_rate]))
                LEDoff_rates[ori] = cluster_ori_LEDoff_rates

        mean_both = pd.concat([LEDon_rates.mean(), LEDoff_rates.mean()], axis=1)
        mean_both.columns = ['LED on', 'LED off']

        max_rate_on[cluster] = mean_both.max()['LED on']
        max_rate_off[cluster] = mean_both.max()['LED off']

        error_both = pd.concat([LEDon_rates.sem(), LEDoff_rates.sem()], axis=1)
        error_both.columns = ['LED on', 'LED off']

        collapsed_means_on[0] = LEDon_rates[180]

        # collapse mean vectors from direction space into orientation space

        for ori in np.arange(45, 180, 45):
            collapsed_means_on[ori] = LEDon_rates[[ori, ori + 180]].mean(axis=1)
        for ori in np.arange(0, 180, 45):
            collapsed_means_off[ori] = LEDoff_rates[[ori, ori + 180]].mean(axis=1)

        collapsed_means_on -= cluster_spike_rate_pre[cluster]
        collapsed_means_off -= cluster_spike_rate_pre[cluster]

        mean_both_collapsed = pd.concat([collapsed_means_on.mean(), collapsed_means_off.mean()], axis=1)
        mean_both_collapsed.columns = ['LED on', 'LED off']

        error_both_collapsed = pd.concat([collapsed_means_on.sem(), collapsed_means_off.sem()], axis=1)
        error_both_collapsed.columns = ['LED on', 'LED off']

        # calculate 1-sample Hotelling T2 test to test if cluster has significant orientation tuning
        # array P holds p-values for all, print which clusters are significant

        alpha = 0.05
        P[cluster] = phy_analysis.t2hot1(collapsed_means_off.as_matrix())

        if P[cluster] >= alpha:
            print('%d mean vectors results not significant, p = %f' % (cluster, P[cluster]))
        else:
            print('%d mean vectors results significant, p = %f' % (cluster, P[cluster]))

        # Use Wilcoxon rank-sum test to calculate if there is a significant difference at each orientation
        T = {}
        w_p = {}
        alpha_bonferroni = 1 - np.power(1 - alpha, 1 / len(collapsed_means_on.columns))

        for ori in collapsed_means_on:
            T[ori], w_p[ori] = sp.stats.wilcoxon(collapsed_means_on[ori].as_matrix(),
                                                 collapsed_means_off[ori].as_matrix())
        w_p_p[cluster] = w_p
        w_p_sig[cluster] = [key for key, value in sorted(w_p.items(), key=lambda x: x[1]) if value < alpha_bonferroni]

        # ori_plot(cluster, LEDon_rates, model, LEDoff_rates, collapsed_means_on,
        #          collapsed_means_off, w_p_sig, P, primary_chan_dict)

    pickle.dump(P, open("hotelling.p", "wb"))
    pickle.dump(w_p_p, open("wilcoxon.p", "wb"))
    pickle.dump(model.cluster_groups, open("model_cluster_groups.p", "wb"))

    os.chdir(beginning_directory)

    return model, primary_chan_dict, P, w_p_sig, max_rate_on, max_rate_off

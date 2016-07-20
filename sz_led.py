import os
from pathlib import Path
import phy_analysis
import analyzer
import tdt_data
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt


def sz_plot(cluster, model, LEDon_rates, LEDoff_rates, mean_on, mean_off, p_perm, primary_chan_dict):

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

    mean_both = pd.concat([mean_on, mean_off], axis=1)
    mean_both.columns = ['LED on', 'LED off']
    error_on, error_off = LEDon_rates.sem(), LEDoff_rates.sem()
    error_both = pd.concat([error_on, error_off], axis=1)
    error_both.columns = ['LED on', 'LED off']

    fig, ax = plt.subplots(1, 1)
    mean_both.plot(linewidth=2, ax=ax)
    ax.fill_between(np.array(LEDon_rates.sem().keys()), LEDon_rates.mean().values - LEDon_rates.sem().values,
                    LEDon_rates.mean().values + LEDon_rates.sem().values,
                    alpha=0.3)

    ax.fill_between(np.array(LEDoff_rates.sem().keys()), LEDoff_rates.mean().values - LEDoff_rates.sem().values,
                    LEDoff_rates.mean().values + LEDoff_rates.sem().values,
                    alpha=0.3, edgecolor='#3F7F4C', facecolor='#7EFF99')

    plt.ylim(ymin=0)
    plt.xlim(0, 100)
    plt.xlabel('Size (degrees)')
    plt.ylabel('Spike Rate (spikes/sec)')
    plt.suptitle('Size Selectivity, Cluster %d' % cluster, fontsize=14, fontweight='bold')
    # plt.title('Manually Sorted: %s, Primary channel: %s, Permutation p-value = %f' %
    #           (model.cluster_groups[cluster], primary_chan_dict[cluster], p_perm))

    os.chdir(model.cluster_groups[cluster])
    plt.savefig("cluster%d.png" % cluster)
    plt.savefig("cluster%d.svg" % cluster, format='svg')

    os.chdir(str(Path(os.getcwd()).parent))
    plt.close(fig)


def suppression_index(preferred_size, largest_stimulus, baseline):
    si = (preferred_size - largest_stimulus) / (preferred_size - baseline)
    return si


def mc_permutation(xs, ys, baseline, num):
    si_shuff = np.zeros(num)
    for perm in range(0, num):
        shuff = np.random.permutation(ys)
        si_shuff[perm] = (shuff.max() - shuff[-1] / (shuff.max()) - baseline)
    return si_shuff


def sz_led(date, block, channel_group, analyzer_path, plot):

    beginning_directory = os.getcwd()
    kwik_path = "C:/Users/anupam/Desktop/salk data/VIPCR/%s/%s/%s.kwik" % (date, block, block)
    tank_path = "C:/Users/anupam/Desktop/salk data/VIPCR/%s/%s" % (date, date)
    trial_num, stim_time = analyzer.looper_led(analyzer_path)

    cluster_ids_dict = phy_analysis.extract_spike_times(kwik_path, channel_group)
    model = phy_analysis.get_kwik_model(kwik_path, channel_group)
    lfpsi, time_index, epocs = tdt_data.extract_ori(tank_path, block)

    # find primary channel of each cluster and save to dictionary
    primary_chan_dict = phy_analysis.primary_channel(kwik_path, block, channel_group)

    # figure out number of samples during pre, post, and stim times
    pre = stim_time[0]
    pre_samples = np.round(pre / lfpsi)  # number of samples in pre-stim time
    post = stim_time[1]
    post_samples = np.round(post / lfpsi)  # number of samples in post-stim time
    stim_length = stim_time[2]
    stim_samples = np.round(stim_length / lfpsi)  # number of samples in stimulus period
    total_length_samples = np.round((pre + stim_length + post) / lfpsi)  # total length in samples

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
    for cluster in cluster_ids_dict:
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
        cluster_spike_trials[cluster] = trial_dict
        cluster_spike_trials_pre[cluster] = pre_trial_dict

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

    # create array of size values used in experiment
    sz = np.unique(trial_num[:, 0])

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

    spike_rates = {}
    si_on_all = {}
    si_on_all_mod = {}
    si_off_all = {}
    p_perm = {}

    for cluster in cluster_ids_dict:

        LEDon_rates = pd.DataFrame()
        LEDon_rates_pre = pd.DataFrame()

        LEDoff_rates = pd.DataFrame()
        LEDoff_rates_pre = pd.DataFrame()

        for size in sz:

            sz_trials = np.where(trial_num[:, 0] == size)
            sz_trials_LEDon = np.where((trial_num[:, 0] == size) & (trial_num[:, 1] == 1))
            sz_trials_LEDoff = np.where((trial_num[:, 0] == size) & (trial_num[:, 1] == 0))

            if np.size(sz_trials_LEDon) > 0:
                cluster_sz_LEDon_rates = pd.Series()
                cluster_sz_LEDon_rates_pre = pd.Series()
                additional = pd.DataFrame()
                additional_pre = pd.DataFrame()
                for trial in np.nditer(sz_trials_LEDon):
                    cluster_sz_LEDon_rate = np.size(cluster_spike_trials[cluster][trial.tolist()]) / stim_length
                    cluster_sz_LEDon_rates = cluster_sz_LEDon_rates.append(pd.Series([cluster_sz_LEDon_rate]))

                    cluster_sz_LEDon_rate_pre = np.size(cluster_spike_trials_pre[cluster][trial.tolist()]) / pre
                    cluster_sz_LEDon_rates_pre = cluster_sz_LEDon_rates_pre.append(
                        pd.Series([cluster_sz_LEDon_rate_pre]))

                additional = pd.DataFrame({size: cluster_sz_LEDon_rates})
                additional_pre = pd.DataFrame({size: cluster_sz_LEDon_rates_pre})

                LEDon_rates = LEDon_rates.append(additional)
                LEDon_rates_pre = LEDon_rates_pre.append(additional_pre)

            if np.size(sz_trials_LEDoff) > 0:
                cluster_sz_LEDoff_rates = pd.Series()
                cluster_sz_LEDoff_rates_pre = pd.Series()
                for trial in np.nditer(sz_trials_LEDoff):
                    cluster_sz_LEDoff_rate = np.size(cluster_spike_trials[cluster][trial.tolist()]) / stim_length
                    cluster_sz_LEDoff_rates = cluster_sz_LEDoff_rates.append(pd.Series([cluster_sz_LEDoff_rate]))

                    cluster_sz_LEDoff_rate_pre = np.size(cluster_spike_trials_pre[cluster][trial.tolist()]) / pre
                    cluster_sz_LEDoff_rates_pre = \
                        cluster_sz_LEDoff_rates_pre.append(pd.Series([cluster_sz_LEDoff_rate_pre]))

                LEDoff_rates[size] = cluster_sz_LEDoff_rates
                LEDoff_rates_pre[size] = cluster_sz_LEDoff_rates_pre

        mean_on, mean_off = LEDon_rates.mean(), LEDoff_rates.mean()
        mean_on_pre, mean_off_pre = LEDon_rates_pre.mean(), LEDoff_rates_pre.mean()
        mean_both_pre = pd.concat([mean_on_pre, mean_off_pre])

        # compute suppression index for both LED states based upon Adesnik et al.
        si_on_all[cluster] = suppression_index(mean_on.max(), mean_on.values[-1], mean_both_pre.mean())
        si_off_all[cluster] = suppression_index(mean_off.max(), mean_off.values[-1], mean_both_pre.mean())

        # modified suppression index using same preferred size as LED off state
        si_on_all_mod[cluster] = suppression_index(mean_on[mean_off.idxmax()], mean_on.values[-1], mean_both_pre.mean())

        # compute permutation test to check for significance of suppression index
        num_perm = 10000
        si_shuff = mc_permutation(mean_off.index.values, mean_off.values, mean_both_pre.mean(), num_perm)
        p_perm[cluster] = np.size(np.where(si_shuff > si_off_all[cluster])) / num_perm

        if plot == 'y':
            sz_plot(cluster, model, LEDon_rates, LEDoff_rates, mean_on, mean_off, p_perm, primary_chan_dict)

    os.chdir(beginning_directory)
    return model, primary_chan_dict, p_perm, si_on_all, si_off_all

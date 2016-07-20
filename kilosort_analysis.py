#!/usr/bin/env python

"""kilosort_analysis.py: functions used for analysis of spike data from KiloSort."""

import os
import datetime
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import analyzer
import phy_analysis


def get_kilo_info(folder_path, sampling_frequency):
    """
    Get all KiloSort information from files produced by KiloSort.
    :param folder_path: path to folder where kilosort files are stored
    :param sampling_frequency: sampling frequency of recording
    :return: pandas DataFrame of all spike information and DataFrame of cluster groups
    """
    print(datetime.datetime.now().strftime("%X") + " Loading KiloSort Data.")

    # Find cluster groups CSV file in folder with all KiloSort data.
    file_name = []
    file_name += [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    file_path = folder_path + '/' + file_name[0]
    cgs = pd.read_csv(file_path, sep="\t").set_index('cluster_id')  # Create DataFrame with cluster groups.

    # Load all spike information based on .npy files in folder into large DataFrame
    spike = pd.DataFrame()
    spike['cluster'] = np.load(folder_path + '/spike_clusters.npy').flatten()  # spike cluster
    spike['sample'] = np.load(folder_path + '/spike_times.npy').flatten()  # sample where spike occurred
    spike['time'] = spike['sample'] / sampling_frequency  # time where spike occurred
    spike['templates'] = np.load(folder_path + '/spike_templates.npy').flatten()  # template used to detect spike
    spike['template_amplitude'] = np.load(folder_path + '/amplitudes.npy').flatten()  # amplitude to scale spike
    spike['cluster_group'] = spike['cluster'].map(cgs['group'].to_dict())  # cluster group of cluster of spike

    # Remove all spikes not counted as "good or "mua"
    spike = spike[(spike['cluster_group'] == 'good') | (spike['cluster_group'] == 'mua')]
    cgs = cgs[(cgs.group != 'noise') & (cgs.group != 'unsorted')]
    return spike, cgs


def get_aux_info(folder_path, num_channels):
    """
    Gets raw auxiliary data from raw binary (.bin) data file.
    :param folder_path: folder path where analyzer file and raw data recording is stored
    :param num_channels: number of auxiliary channels in recording
    :return:
        data: pandas DataFrame of all raw auxiliary data
        all_exp_times: experiment data and timing from analyzer files and epoc/stimulus pulses
        stim_time: numpy ndarray of pre-stim, stim, and post-stim times
    """
    # Find .bin file and load into numpy memory-mapped array to conserve system memory.
    file_name = []
    file_name += [file for file in os.listdir(folder_path) if file.endswith('.bin')]
    file_path = folder_path + '/' + file_name[0]

    data = pd.DataFrame()
    print(datetime.datetime.now().strftime("%X") + " Loading Raw Data.")
    data_array = (np.memmap(file_path, np.int16, mode='r').reshape(-1,  128 + num_channels)).T

    # Build DataFrame of auxiliary channel raw data
    print(datetime.datetime.now().strftime("%X") + " Building DataFrame of Auxiliary Channels.")
    # aux_labels = ['epoc', 'photodiode', 'movement-A', 'movement-B', 'LED']
    aux_labels = ['photodiode', 'epoc', 'stimulus', 'movement-A', 'movement-B', 'LED']  # note order of aux channels!

    for i, label in enumerate(aux_labels):
        data[label] = data_array[-(num_channels-i), :]
        print(datetime.datetime.now().strftime("%X") + " Loaded %s." % label)

    del data_array  # remove memory-mapped array from system memory!

    print(datetime.datetime.now().strftime("%X") + ' Extracting trial timing information.')
    # Look for pulses in epoc and stimulus channels and store start and end times in timing DataFrame
    timing = {}
    timing_df = pd.DataFrame()
    for channel in ['epoc', 'stimulus']:
        data['tag'] = data[channel] > 1000
        # first row is a True preceded by a False
        fst = data.index[data['tag'] & ~ data['tag'].shift(1).fillna(False)]
        # last row is a True followed by a False
        lst = data.index[data['tag'] & ~ data['tag'].shift(-1).fillna(False)]
        # filter those which are adequately apart
        timing[channel] = np.asarray([(i, j) for i, j in zip(fst, lst) if j > i + 4])

        timing_df[(channel + '_start')] = np.asarray(fst)
        timing_df[(channel + '_end')] = np.asarray(lst)

    # Get trial condition information from analyzer file used in experiment (stored in same folder)
    analyzer_path = analyzer.get_analyzer_path(folder_path)
    trial_num, stimulus_time = analyzer.looper_led_blanks(analyzer_path)
    trial_num_df = pd.DataFrame(trial_num, columns=['stim_presented', 'light_bit'])

    # Combine analyzer data with trial start and end times
    all_exp_times = pd.concat([trial_num_df, timing_df], axis=1)

    return data, all_exp_times, stimulus_time


def get_condition_data(sp_info, tri_timing):
    """

    :param sp_info: pandas DataFrame of all spiking info from kilosort_analysis.get_kilo_info
    :param tri_timing: pandas DataFrame of trial timing information from kilosort_analysis.get_aux_info
    :return:
        tri_timing: pandas dataframe of trial timing with extra columns for spike count of each cluster during each
        trial
    """
    # Get spike counts for each cluster during each condition. Vectorized for speed/efficiency!!
    num_clusters = len(np.unique(sp_info['cluster'].values))
    for i, cluster in enumerate(np.unique(sp_info['cluster'].values)):
        print(datetime.datetime.now().strftime("%X") + " Processing Cluster %d of %d" % (i + 1, num_clusters))
        sp_info_clust = sp_info[sp_info['cluster'] == cluster]
        all_times = np.sort(np.append(tri_timing['stimulus_start'].values, tri_timing['stimulus_end'].values))
        tri_timing['cluster_' + str(cluster)] = sp_info_clust['sample'].groupby(pd.cut(sp_info_clust['sample'],
                                                                                       all_times)).count()[::2].values

    return tri_timing


def cluster_ori_led_plot(cgs, timing_counts, folder_path, sampling_frequency):
    """

    :param cgs: spike information returned by get_kilo_info
    :param timing_counts: pandas DataFrame of experiment results sorted by condition returned from get_condition_data
    :param folder_path: path to folder where figures are to be stored
    :param sampling_frequency: sampling frequency of recording
    :return:
        plots all stim_presented tuning curves by LED condition and saves in folder
    """
    if not os.path.exists(folder_path + '/figures'):
        os.makedirs(folder_path + '/figures')

    p, max_rates_on, max_rates_off = (np.zeros(np.size(cgs.index.values)) for i in range(0, 3))
    plt.ioff()  # Turn off interactive plotting so plot windows do not automatically open
    for i, cluster in enumerate(cgs.index.values):  # Try to vectorize to improve speed?
        print(datetime.datetime.now().strftime("%X") + ' Plotting Cluster %d.' % cluster)
        fig, ax = plt.subplots(1, 1)
        # Create new DataFrame with just information from relevant cluster
        plot_df = timing_counts[['stim_presented', 'light_bit', 'cluster_' + str(cluster)]]
        plot_df['trial_length'] = timing_counts['stimulus_end'] - timing_counts['stimulus_start']
        plot_df['spike_rate'] = plot_df['cluster_' + str(cluster)] / (plot_df['trial_length'] / sampling_frequency)

        baseline = plot_df[plot_df.stim_presented == 256].groupby('light_bit')['spike_rate'].mean().to_dict()
        plot_df = plot_df[plot_df.stim_presented != 256]  # remove blanks from plotting data

        plot_df['baseline'] = plot_df['light_bit'].map(baseline)
        plot_df['spike_rate'] = plot_df.spike_rate - plot_df.baseline

        plot_df.drop(['trial_length', 'cluster_' + str(cluster), 'baseline'], inplace=True, axis=1)

        plot_df['count'] = plot_df.groupby(['stim_presented', 'light_bit']).cumcount()
        hotelling_p = phy_analysis.t2hot1(plot_df[plot_df.light_bit == 0].pivot(
            'count', 'stim_presented', 'spike_rate').as_matrix())
        p[i] = hotelling_p
        plot_df.drop(['count'], inplace=True, axis=1)

        means = plot_df.groupby(["stim_presented", "light_bit"]).mean().unstack(level=1)
        errors = plot_df.groupby(["stim_presented", "light_bit"]).sem().unstack(level=1)

        max_rates_off[i] = means.max()[0]
        max_rates_on[i] = means.max()[1]

        means.plot(yerr=errors, ax=ax, linewidth=3)

        plt.suptitle('Cluster %d Orientation Tuning' % cluster)
        ax.set_title('P = %f' % hotelling_p)
        plt.xlabel('Orientation (degrees)')
        plt.ylabel('Spike Rate (spikes/sec)')
        plt.savefig(folder_path + '/figures/cluster%d.png' % cluster)
        plt.savefig(folder_path + '/figures/cluster%d.eps' % cluster, format='eps')
        plt.close()
    cgs['hotelling_p'] = p
    cgs['max_rates_on'] = max_rates_on
    cgs['max_rates_off'] = max_rates_off

    plt.ion()

    return cgs


def plot_max_rates(cgs):
    params = {
        'axes.labelsize': 8,
        'text.fontsize': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [4.5, 4.5]
    }
    rcParams.update(params)

    # plot max rates for each LED state for clusters that are significantly orientation tuned
    fig, ax = plt.subplots()
    x_max = cgs[cgs.hotelling_p < 0.05]['max_rates_off']
    y_max = cgs[cgs.hotelling_p < 0.05]['max_rates_on']
    ax.scatter(x_max, y_max, label='_nolegend_')

    # Plot best-fit line of scatter data above
    ax.plot(x_max, np.poly1d(np.polyfit(x_max, y_max, 1))(x_max), label='Line of Best Fit')

    # compute Wilcoxon signed-rank p-value
    wilcoxon_T, wilcoxon_p = sp.stats.wilcoxon(x_max, y_max)

    # Plot unity line on same plot
    ax.autoscale(False)  # Turn off auto-scaling so unity line does not affect axis of plot
    unity = np.arange(-25, 25, 0.01)
    ax.plot(unity, unity, '--', label='Unity Line')

    plt.suptitle('Comparison of Maximum Firing Rates for Tuned Cells')
    plt.title('p-value: %f' % wilcoxon_p)
    plt.xlabel('Max Rate Off (spikes/sec)')
    plt.ylabel('Max Rate On (spikes/sec)')
    legend = ax.legend()
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    plt.show()


def cluster_sz_led_plot(cgs, timing_counts, folder_path, sampling_frequency):
    """

    :param cgs: spike information returned by get_kilo_info
    :param timing_counts: pandas DataFrame of experiment results sorted by condition returned from get_condition_data
    :param folder_path: path to folder where figures are to be stored
    :param sampling_frequency: sampling frequency of recording
    :return:
        plots all stim_presented tuning curves by LED condition and saves in folder
    """
    if not os.path.exists(folder_path + '/figures'):
        os.makedirs(folder_path + '/figures')

    plt.ioff()
    for cluster in cgs.index.values:
        print(datetime.datetime.now().strftime("%X") + ' Plotting Cluster %d.' % cluster)
        fig, ax = plt.subplots(1, 1)
        plot_df = timing_counts[['stim_presented', 'light_bit', 'cluster_' + str(cluster)]]
        plot_df['trial_length'] = timing_counts['stimulus_end'] - timing_counts['stimulus_start']
        plot_df['spike_rate'] = plot_df['cluster_' + str(cluster)] / (plot_df['trial_length'] / sampling_frequency)

        baseline = plot_df[plot_df.stim_presented == 256].groupby('light_bit')['spike_rate'].mean().to_dict()
        plot_df = plot_df[plot_df.stim_presented != 256]  # remove blanks from plotting data

        plot_df['baseline'] = plot_df['light_bit'].map(baseline)
        plot_df['spike_rate'] = plot_df.spike_rate - plot_df.baseline

        plot_df.drop(['trial_length', 'cluster_' + str(cluster), 'baseline'], inplace=True, axis=1)

        means = plot_df.groupby(["stim_presented", "light_bit"]).mean().unstack(level=1)
        errors = plot_df.groupby(["stim_presented", "light_bit"]).sem().unstack(level=1)

        means.plot(yerr=errors, ax=ax, linewidth=3)

        plt.suptitle('Cluster %d Size Tuning' % cluster)
        plt.xlabel('Size (degrees)')
        plt.ylabel('Spike Rate (spikes/sec)')
        plt.savefig(folder_path + '/figures/cluster%d.png' % cluster)
        plt.savefig(folder_path + '/figures/cluster%d.eps' % cluster, format='eps')
        plt.close()
    plt.ion()


def plot_raw(data_array):

    from matplotlib.offsetbox import AnchoredOffsetbox

    class AnchoredScaleBar(AnchoredOffsetbox):
        def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                     pad=0.1, borderpad=0.1, sep=2, prop=None, **kwargs):
            """
            Draw a horizontal and/or vertical  bar with the size in data coordinate
            of the give axes. A label will be drawn underneath (center-aligned).
            - transform : the coordinate frame (typically axes.transData)
            - sizex,sizey : width of x,y bar, in data units. 0 to omit
            - labelx,labely : labels for x,y bars; None to omit
            - loc : position in containing axes
            - pad, borderpad : padding, in fraction of the legend font size (or prop)
            - sep : separation between labels and bars in points.
            - **kwargs : additional arguments passed to base class constructor
            """
            from matplotlib.patches import Rectangle
            from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
            bars = AuxTransformBox(transform)
            if sizex:
                bars.add_artist(Rectangle((0, 0), sizex, 0, fc="none"))
            if sizey:
                bars.add_artist(Rectangle((0, 0), 0, sizey, fc="none"))

            if sizex and labelx:
                bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False)],
                               align="center", pad=0, sep=sep)
            if sizey and labely:
                bars = HPacker(children=[TextArea(labely), bars],
                               align="center", pad=0, sep=sep)

            AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                       child=bars, prop=prop, frameon=False, **kwargs)

    def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
        """ Add scalebars to axes
        Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
        and optionally hiding the x and y axes
        - ax : the axis to attach ticks to
        - matchx,matchy : if True, set size of scale bars to spacing between ticks
                        if False, size should be set using sizex and sizey params
        - hidex,hidey : if True, hide x-axis and y-axis of parent
        - **kwargs : additional arguments passed to AnchoredScaleBars
        Returns created scalebar object
        """

        def f(axis):
            l = axis.get_majorticklocs()
            return len(l) > 1 and (l[1] - l[0])

        if matchx:
            kwargs['sizex'] = f(ax.xaxis)
            kwargs['labelx'] = str(kwargs['sizex'])
        if matchy:
            kwargs['sizey'] = f(ax.yaxis)
            kwargs['labely'] = str(kwargs['sizey'])

        sb = AnchoredScaleBar(ax.transData, **kwargs)
        ax.add_artist(sb)

        if hidex: ax.xaxis.set_visible(False)
        if hidey: ax.yaxis.set_visible(False)

        return sb

    mapping = {
    2: 103,
    3: 39,
    4: 104,
    5: 41,
    6: 102,
    7: 38,
    8: 105,
    9: 42,
    10: 101,
    11: 37,
    12: 106,
    13: 43,
    14: 100,
    15: 36,
    16: 107,
    17: 44,
    19: 35,
    20: 108,
    21: 45,
    22: 98,
    23: 34,
    24: 109,
    25: 46,
    26: 97,
    27: 33,
    28: 110,
    29: 47,
    30: 96,
    31: 32,
    32: 111,
    34: 127,
    35: 63,
    36: 112,
    37: 49,
    38: 126,
    39: 62,
    40: 113,
    41: 50,
    42: 125,
    43: 61,
    44: 114,
    45: 51,
    46: 124,
    47: 60,
    48: 115,
    49: 52,
    51: 59,
    52: 123,
    53: 53,
    54: 117,
    55: 58,
    56: 122,
    57: 54,
    58: 118,
    59: 57,
    60: 121,
    61: 55,
    62: 119,
    63: 56,
    64: 120,
    66: 71,
    67: 7,
    68: 72,
    69: 9,
    70: 70,
    71: 6,
    72: 73,
    73: 10,
    74: 74,
    75: 5,
    76: 69,
    77: 11,
    78: 75,
    79: 4,
    80: 68,
    81: 12,
    83: 3,
    84: 67,
    85: 13,
    86: 77,
    87: 2,
    88: 66,
    89: 14,
    90: 78,
    91: 1,
    92: 65,
    93: 15,
    94: 79,
    95: 0,
    96: 64,
    98: 80,
    99: 31,
    100: 95,
    101: 17,
    102: 81,
    103: 30,
    104: 94,
    105: 18,
    106: 82,
    107: 29,
    108: 93,
    109: 19,

    }

    # data scaling
    Vdd = 2             # ADC supply range
    Vss = -2
    ADC_bits = 16       # bits of ADC
    gain_neural = 200   # gain of Intan
    gain_dig_aux = 1/4  # gain of digital aux channels

    scale_neural = ((Vdd-Vss)/(np.power(2, ADC_bits)))/gain_neural

    fig, ax = plt.subplots(1, 1)

    for i in mapping:
        plt.plot((data_array[mapping[i], 608267+7000:608267+12000] + i * 100) * scale_neural)

    add_scalebar(ax, matchx=False, matchy=False, sizex=500, sizey=0.0005)

    plt.show()
    plt.savefig('raw.eps', format='eps')

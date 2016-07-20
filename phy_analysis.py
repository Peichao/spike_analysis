import os
import scipy as sp
from scipy import stats
import numpy as np
import tkinter as tk
from tkinter import filedialog
from klusta.kwik.model import KwikModel


def get_kwik_path():
    """
    gets path to kwik file from file dialog
    :return: path to kwik file
    """
    tk.Tk().withdraw()
    kwik_path = tk.filedialog.askopenfilename(filetypes=[("kwik files", "*.kwik")])
    return kwik_path


def create_prms(save_path, filename, probe):
    """
    Creates parameters file for spikesorting data in phy.
    :param save_path: string, path to save parameters file.
    :param filename: string, filename of dat file to be sorted (*.dat)
    :param probe: string, probe file to be used (ex: a2x16)
    :return: new parameter file created in indicated directory.
    """
    f = open('phy_sample_parameters.prm', 'r')
    temp = f.read()
    f.close()

    f = open(save_path, 'w')
    f.write("experiment_name = '%s'\nprb_file = '%s'\n\n" % (filename, probe))
    f.write(temp)
    f.close()


def spikesort(prm_path):
    """
    Sorts data in phy with input parameters file.
    :param prm_path: string of path to phy parameters file
    :return: sorted data in phy
    """
    prm_path = prm_path
    current_directory = os.getcwd()
    spikesort_directory = os.path.dirname(prm_path)
    os.chdir(spikesort_directory)
    os.system('phy spikesort "%s"' % prm_path)
    os.chdir(current_directory)


def extract_spike_times(kwik_path, channel_group):
    """
    takes manually sorted kwik file, and returns a dictionary of all clusters as keys and their spike times
    :param kwik_path: string path to the kwik file to be analyzed
    :param channel_group: shank number to be sorted
    :return: dictionary, keys = clusters, values = list of spike times of that cluster
    """
    model = KwikModel(kwik_path, channel_group=channel_group)
    model.clustering = 'main'
    model.describe()

    spike_times = model.spike_times
    spike_clusters = model.spike_clusters
    cluster_ids = model.cluster_ids
    cluster_ids_dict = {key: [] for key in cluster_ids}
    for spike_time_idx in range(0, len(spike_times)):
        cluster_ids_dict[spike_clusters[spike_time_idx]].append(spike_times[spike_time_idx])
    return cluster_ids_dict


def get_kwik_model(kwik_path, channel_group):
    """
    Returns Kwik model to be analyzed in python.
    :param kwik_path: string to manually sorted *.kwik file
    :param channel_group: shank number to be sorted
    :return: phy model
    """
    model = KwikModel(kwik_path, channel_group=channel_group)
    model.clustering = 'main'
    return model


def primary_channel(kwik_path, block, channel_group):
    kwik_dir = os.path.abspath(os.path.join(kwik_path, os.pardir))
    phy_dir = kwik_dir + '/' + block + '.phy'
    phymetadir = phy_dir + '/cluster_store/%d/main/' % channel_group
    filesphy = [(phymetadir, filename) for filename in os.listdir(phymetadir)]

    have_mask = [file_entry[1] for file_entry in filesphy if '.mean_masks' in file_entry[1]]

    cluster_id = np.zeros((len(have_mask)))
    primary_chan = np.zeros((len(have_mask)))

    for idx, mask_file in enumerate(have_mask):
        waveformsfile = phymetadir + mask_file
        arr = np.fromfile(waveformsfile, dtype=np.float32)

        cluster_id[idx] = mask_file.split('.')[0]
        primary_chan[idx] = np.argmax(arr)

    makedict = {'cluster_id': cluster_id, 'primary_chan': primary_chan}
    keys = makedict['cluster_id'].astype(int)
    values = makedict['primary_chan'].astype(int)
    primary_chan_dict = dict(zip(keys, values))
    return primary_chan_dict


def t2hot1(X):
    n = np.shape(X)[0]
    p = np.shape(X)[1]
    mu = np.zeros([1, p])
    m = np.mean(X, axis=0)
    S = np.cov(X.T)
    if sp.linalg.det(S) == 0:
        return 1
    else:
        T2 = np.dot(np.dot((n*(m-mu)), sp.linalg.inv(S)), (m - mu).conj().transpose())

        if n >=50:
            X2 = T2
            v = p
            P = 1 - sp.stats.chi2.cdf(X2, v)

        else:
            F = (n - p) / ((n - 1) * p) * T2
            v1 = p
            v2 = n - p
            P = 1 - sp.stats.f.cdf(F, v1, v2)

    return P
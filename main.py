# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-11-08 14:32:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-12-05 14:22:49
from kilosort import run_kilosort, DEFAULT_SETTINGS
from kilosort.io import load_probe
import os, sys
import numpy as np
from functions import loadXML
from phy_to_klusters import phy_to_klusters
import shutil

if os.path.exists("/mnt/Data/Data/"):
    data_directory = "/mnt/Data/Data"
elif os.path.exists('/mnt/DataRAID2/'):    
    data_directory = '/mnt/DataRAID2/'
elif os.path.exists('/mnt/ceph/users/gviejo'):    
    data_directory = '/mnt/ceph/users/gviejo'
elif os.path.exists('/media/guillaume/Raid2'):
    data_directory = '/media/guillaume/Raid2'

datasets = np.hstack([
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
    ])


# datasets = yaml.safe_load(open("/mnt/ceph/users/gviejo/datasets_OPTO.yaml", "r"))
# datasets = datasets['opto']['opto_adn_psb']['sleep'] + datasets['opto']['opto_lmn_psb']['sleep']






datasets = np.unique(datasets)

# for s in datasets:
# for s in ['LMN/A1411/A1411-200907A']:
# for s in ['LMN/A6701/A6701-201207A']:
for s in ["LMN-ADN/A5010/A5010-201003A","LMN-ADN/A5002/A5002-200304A","LMN-ADN/A5002/A5002-200306A","LMN-ADN/A5043/A5043-230303A","LMN-ADN/A5044/A5044-240329A"]:

    # path = os.path.join(data_directory, "OPTO", s)
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)

    #################################################################
    # parameters to load the xml
    num_channels, fs, shank_to_channel, shank_to_keep = loadXML(path)


    #################################################################
    # Kilosort settings
    settings = DEFAULT_SETTINGS
    settings['n_chan_bin'] = num_channels
    settings['fs'] = fs
    settings['filename'] =  os.path.join(path, os.path.basename(path) + ".dat")
    settings['batch_size'] = 12*fs
    # settings['nt'] = 41 # This should always be odd.
    settings['nearest_chans'] = 8
    settings['nearest_templates'] = np.max([len(shank_to_channel[i]) for i in shank_to_channel.keys()])
    settings['Th_universal'] = 12

    probe = load_probe(os.path.join(path, "chanMap.mat"))

    probe['xc'] = probe['xc'] * 100

    ################################################################
    # Kilosort run
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(settings=settings, probe = probe)


    ################################################################
    # Export to neurosuite

    savepath = os.path.join(path, "kilosort4")

    shutil.copyfile(os.path.join(path, basename+".xml"), os.path.join(savepath, basename+".xml"))

    phy_to_klusters(path, savepath, probe)

    ################################################################
    # Clean npy and tsv files

    for f in ['phy.log','cluster_KSLabel.tsv','whitening_mat.npy','channel_shanks.npy','spikes_ks4.npz','amplitudes.npy','spike_templates.npy','spike_positions.npy','cluster_Amplitude.tsv','kilosort4.log','cluster_group.tsv','templates.npy','pc_features.npy','whitening_mat_dat.npy','channel_positions.npy','spike_times.npy','similar_templates.npy','spike_clusters.npy','channel_map.npy','spike_detection_templates.npy','params.py','pc_feature_ind.npy','ops.npy','kept_spikes.npy','templates_ind.npy','whitening_mat_inv.npy','cluster_ContamPct.tsv']:

        try:
            os.remove(os.path.join(savepath, f))
        except Exception:
            pass
    

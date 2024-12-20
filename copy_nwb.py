# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-11-08 14:32:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-12-17 17:05:02
# %%
import os, sys
import numpy as np
from functions import loadXML
from phy_to_klusters import phy_to_klusters
import shutil
import pandas as pd
from pathlib import Path



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
    np.genfromtxt(os.path.join(data_directory,'datasets_ADN.list'), delimiter = '\n', dtype = str, comments = '#'),
    np.genfromtxt(os.path.join(data_directory,'datasets_LMN_ADN.list'), delimiter = '\n', dtype = str, comments = '#')
    ])


# datasets = yaml.safe_load(open("/mnt/ceph/users/gviejo/datasets_OPTO.yaml", "r"))
# datasets = datasets['opto']['opto_adn_psb']['sleep'] + datasets['opto']['opto_lmn_psb']['sleep']


target_folder = "/run/media/gviejo/GuillaumeTrash"

# %%

datasets = np.unique(datasets)

for s in datasets:

    # path = os.path.join(data_directory, "OPTO", s)
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)

    ks4_path = os.path.join(path, "kilosort4")
    og_path = os.path.join(path, "pynapplenwb")

    filepath = os.path.join(ks4_path, basename+".nwb")

    if os.path.exists(filepath):

        target_path = os.path.join(target_folder, s, "kilosort4")

        target_filepath = os.path.join(target_path, basename+".nwb")

        path_way = Path(target_path)

        path_way.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(filepath, target_filepath)
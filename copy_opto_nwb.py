# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-11-08 14:32:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-12-18 12:25:37
# %%
import os, sys
import numpy as np
from functions import loadXML
from phy_to_klusters import phy_to_klusters
import shutil
import pandas as pd
from pathlib import Path
import yaml


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


datasets = yaml.safe_load(open("/mnt/ceph/users/gviejo/datasets_OPTO.yaml", "r"))

tmp = []
for a in datasets['opto'].keys():
    for e in ['wake', 'sleep']:        
        x = datasets['opto'][a][e]
        for s in x:
            if s != None:
                tmp.append(s)
tmp = np.array(tmp)


#datasets = datasets['opto']['opto_adn_psb']['sleep'] + datasets['opto']['opto_lmn_psb']['sleep']
datasets=tmp


target_folder = "/run/media/gviejo/GuillaumeTrash/OPTO"

# %%

datasets = np.unique(datasets)

for s in datasets:

    path = os.path.join(data_directory, "OPTO", s)    
    basename = os.path.basename(path)

    # ks4_path = os.path.join(path, "kilosort4")
    og_path = os.path.join(path, "pynapplenwb")

    filepath = os.path.join(og_path, basename+".nwb")

    if os.path.exists(filepath):
        
        target_path = os.path.join(target_folder, s, "kilosort4")

        target_filepath = os.path.join(target_path, basename+".nwb")

        path_way = Path(target_path)

        path_way.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(filepath, target_filepath)
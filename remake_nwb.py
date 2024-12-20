# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-11-08 14:32:37
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-12-12 15:40:50
# %%
import os, sys
import numpy as np
from functions import loadXML
from phy_to_klusters import phy_to_klusters
import shutil
import nwbmatic as ntm
from pynwb import NWBHDF5IO
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import CompassDirection, Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
import pynapple as nap
import datetime
import pandas as pd
from pathlib import Path

def load_neurosuite_spikes(path, basename, time_support=None, fs=20000.0):
    files = os.listdir(path)
    clu_files = np.sort([f for f in files if ".clu." in f and f[0] != "."])
    res_files = np.sort([f for f in files if ".res." in f and f[0] != "."])
    clu1 = np.sort([int(f.split(".")[-1]) for f in clu_files])
    clu2 = np.sort([int(f.split(".")[-1]) for f in res_files])
    if len(clu_files) != len(res_files) or not (clu1 == clu2).any():
        raise RuntimeError(
            "Not the same number of clu and res files in " + path + "; Exiting ..."
        )

    count = 0
    spikes = {}
    group = pd.Series(dtype=np.int32)
    for i, s in zip(range(len(clu_files)), clu1):
        clu = np.genfromtxt(
            os.path.join(path, basename + ".clu." + str(s)), dtype=np.int32
        )[1:]
        if np.max(clu) > 1:  # getting rid of mua and noise
            res = np.genfromtxt(os.path.join(path, basename + ".res." + str(s)))
            tmp = np.unique(clu).astype(int)
            idx_clu = tmp[tmp > 1]
            idx_out = np.arange(count, count + len(idx_clu))

            for j, k in zip(idx_clu, idx_out):
                t = res[clu == j] / fs
                spikes[k] = nap.Ts(t=t, time_units="s")
                group.loc[k] = s

            count += len(idx_clu)

    group = group - 1  # better to start it a 0

    spikes = nap.TsGroup(
        spikes, time_support=time_support, time_units="s", group=group
    )

    return spikes

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




# %%

datasets = np.unique(datasets)

for s in datasets:
# for s in ['LMN-ADN/A5011/A5011-201014A']:
# for s in ['LMN/A1411/A1411-200907A']:
# for s in ['LMN/A6701/A6701-201207A']:


    # path = os.path.join(data_directory, "OPTO", s)
    path = os.path.join(data_directory, s)
    basename = os.path.basename(path)

    ks4_path = os.path.join(path, "kilosort4")
    og_path = os.path.join(path, "pynapplenwb")

    if os.path.exists(ks4_path) and os.path.exists(og_path):

        data = ntm.load_session(path, 'neurosuite')
        sws_ep = data.read_neuroscope_intervals('sws')
        rem_ep = data.read_neuroscope_intervals('rem')        
        
        data = nap.load_file(os.path.join(path, "pynapplenwb", basename+".nwb"))
        
        num_channels, fs, group_to_channel, shank_to_keep = loadXML(path)

        nwbfilepath = os.path.join(
            ks4_path, basename + ".nwb"
        )

        nwbfile = NWBFile(
            session_description=data.nwb.session_description,
            identifier=data.nwb.identifier,
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
            experimenter="Guillaume Viejo, Sofia Skromne Carrasco",
            lab="Adrien Peyrache lab",
            institution="Montreal Neurological Institute, McGill",
            subject=Subject(sex="M", species="Mus Musculus", strain="C57BL/6J"),
        )

        # Tracking
        position = Position()
        for c in ["x", "y", "z"]:
            tmp = SpatialSeries(
                name=c,
                data=data[c].values[:],
                timestamps=data[c].index.values,
                unit="cm",
                reference_frame="",
            )
            position.add_spatial_series(tmp)
        direction = CompassDirection()
        for c in ["rx", "ry", "rz"]:
            tmp = SpatialSeries(
                name=c,
                data=data[c].values[:],
                timestamps=data[c].index.values,
                unit="radian",
                reference_frame="",
            )
            direction.add_spatial_series(tmp)

        nwbfile.add_acquisition(position)
        nwbfile.add_acquisition(direction)

        # INtervals
        position_time_support = TimeIntervals(
            name="position_time_support",
            description="The time support of the position i.e the real start and end of the tracking",
            )
        epochs = data['position_time_support']
        for i, s, e in zip(range(len(epochs)), epochs.start, epochs.end):
            position_time_support.add_interval(start_time=s, stop_time=e, tags=str(i))

        nwbfile.add_time_intervals(position_time_support)

        beh_epochs = TimeIntervals(
            name="epochs",
            description="Sleep and wake epochs")
        epochs = data['epochs']
        for i, s, e, t in zip(range(len(epochs)), epochs.start, epochs.end, epochs.tags):                        
            beh_epochs.add_interval(
                start_time=s,
                stop_time=e,
                tags=t[0])
        nwbfile.add_time_intervals(beh_epochs)

        rem_epochs = TimeIntervals(
            name="rem",
            description="REM sleep")
        epochs = rem_ep
        for i, s, e in zip(range(len(epochs)), epochs.start, epochs.end):
            rem_epochs.add_interval(
                start_time=s,
                stop_time=e,
                tags=str(i))
        nwbfile.add_time_intervals(rem_epochs)

        sws_epochs = TimeIntervals(
            name="sws",
            description="NREM sleep")
        epochs = sws_ep
        for i, s, e in zip(range(len(epochs)), epochs.start, epochs.end):                        
            sws_epochs.add_interval(
                start_time=s,
                stop_time=e,
                tags=str(i))
        nwbfile.add_time_intervals(sws_epochs)
                
        # Units
        spikes = load_neurosuite_spikes(ks4_path, basename, time_support=None, fs=20000.0)

        # Need to intersect metadata
        metadata = data['units'][['location','group']]
        group_to_location = {}
        for g in group_to_channel.keys():
            if g in metadata.group.values:
                group_to_location[g] = np.unique(metadata.location[metadata.group==g])[0]
            else:
                group_to_location[g] = "-"

        groups = spikes.group
        locations = [group_to_location[g] for g in groups]


        electrode_groups = {}

        for g in group_to_channel:
            print(g)
            device = nwbfile.create_device(
                name="shank-" + str(g),
                description="",
                manufacturer="",
            )

            electrode_groups[g] = nwbfile.create_electrode_group(
                name="group" + str(g),
                description="",
                position=(0,0,0),
                location=group_to_location[g],
                device=device,
            )

            for idx in group_to_channel[g]:
                nwbfile.add_electrode(
                    id=idx,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    imp=0.0,
                    location=group_to_location[g],
                    filtering="none",
                    group=electrode_groups[g],
                )

        meanwf = {}
        stdwf = {}

        # Loading waveforms
        for shank in np.unique(spikes.group):
            spk_file = os.path.join(path, "kilosort4", basename + '.spk.%i' %(shank+1))
            res_file = os.path.join(path, "kilosort4", basename + '.res.%i' %(shank+1))
            clu_file = os.path.join(path, "kilosort4", basename + '.clu.%i' %(shank+1))
            res = np.genfromtxt(res_file, dtype="int")
            clu = np.genfromtxt(clu_file, dtype="int")[1:]
            clu_index = np.unique(clu)
            clu_index = clu_index[clu_index>1]
            array = np.fromfile(spk_file, dtype=np.int16)
            array = array.reshape((
                len(res), 32, len(group_to_channel[shank])
                ))

            for i,n in zip(spikes[spikes.group==shank].index, clu_index):
                meanwf[i] = np.mean(array[clu==n], 0)
                stdwf[i] = np.std(array[clu==n], 0)

        # NWB can't take waveforms if number of channels don't match
        max_channels = np.max([meanwf[i].shape[1] for i in meanwf.keys()])

        for k in meanwf.keys():
            meanwf[k] = np.hstack((meanwf[k], np.ones((32, max_channels-meanwf[k].shape[1]))*np.nan))
            stdwf[k] = np.hstack((stdwf[k], np.ones((32, max_channels-stdwf[k].shape[1]))*np.nan))


        # Adding units
        nwbfile.add_unit_column("location", "the anatomical location of this unit")
        nwbfile.add_unit_column("group", "the group of the unit")
        for i, u in enumerate(spikes.keys()):
            nwbfile.add_unit(
                id=u,
                spike_times=spikes[u].as_units("s").index.values,
                electrode_group=electrode_groups[groups[i]],
                location=locations[i],
                group=groups[i],
                waveform_mean = meanwf[u],
                waveform_sd = stdwf[u]
            )

            

        # Writing
        with NWBHDF5IO(nwbfilepath, "w") as io:
            io.write(nwbfile)
            io.close()


# %%
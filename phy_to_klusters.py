import numpy as np
import pandas as pd
from pylab import *
import sys, os
import pynapple as nap
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def loadXML(path):
    """
    path should be the folder session containing the XML file
    Function returns :
        1. the number of channels
        2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
            eeg file first if both are present or both are absent
        3. the mappings shanks to channels as a dict
    Args:
        path : string

    Returns:
        int, int, dict
    """
    if not os.path.exists(path):
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()
    listdir = os.listdir(path)
    xmlfiles = [f for f in listdir if f.endswith('.xml')]
    if not len(xmlfiles):
        print("Folder contains no xml files; Exiting ...")
        sys.exit()
    new_path = os.path.join(path, xmlfiles[0])
    
    from xml.dom import minidom 
    xmldoc      = minidom.parse(new_path)
    nChannels   = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
    fs_dat      = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
    fs_eeg      = xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data  
    if os.path.splitext(xmlfiles[0])[0] +'.dat' in listdir:
        fs = fs_dat
    elif os.path.splitext(xmlfiles[0])[0] +'.eeg' in listdir:
        fs = fs_eeg
    else:
        fs = fs_eeg
    shank_to_channel = {}
    groups      = xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
    for i in range(len(groups)):
        shank_to_channel[i] = np.array([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
    return int(nChannels), int(fs), shank_to_channel


def get_memory_map(filepath, nChannels, frequency=20000):
    """Summary
    
    Args:
        filepath (TYPE): Description
        nChannels (TYPE): Description
        frequency (int, optional): Description
    """
    n_channels = int(nChannels)    
    f = open(filepath, 'rb') 
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2      
    n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
    duration = n_samples/frequency
    interval = 1/frequency
    f.close()
    fp = np.memmap(filepath, np.int16, 'r', shape = (n_samples, n_channels))        
    timestep = np.arange(0, n_samples)/frequency

    return fp, timestep

def phy_to_klusters(path, savepath, probe, n_t = 32):
    basename = os.path.basename(path)
    basepath = savepath
    filepath = os.path.join(path, basename+'.dat')
    n_channels, fs, shank_to_channel = loadXML(path)

    #
    spike_times = np.load(os.path.join(savepath, "spike_times.npy"))
    spike_clusters = np.load(os.path.join(savepath, "spike_clusters.npy"))        
    templates = np.load(os.path.join(savepath, "templates.npy"))

    # Need to determine shank position manually. 
    spike_positions = np.load(os.path.join(savepath, "spike_positions.npy"))
    channel_positions = np.load(os.path.join(savepath, "channel_positions.npy"))
    channel_shanks = np.load(os.path.join(savepath, "channel_shanks.npy"))

    xcenters = np.array([np.mean(channel_positions[channel_shanks==i,0]) for i in np.unique(channel_shanks)])
    d = np.diff(xcenters)
    bins = np.hstack((
        xcenters[[0]]-np.max(d),
        xcenters[0:-1] + d/2,
        xcenters[[-1]] + np.max(d)))

    groups = np.digitize(spike_positions[:,0], bins)-1

    shanks = np.unique(probe['kcoords']).astype("int")

    # Batching of lfp
    fp, timestep = get_memory_map(filepath, n_channels, frequency=fs)
    n = len(fp)
    n_batches = int(np.ceil(n/(20*60*fs)))
    batch_size = int(n/n_batches)
    starts = np.arange(0, n, batch_size, dtype=np.int64)

    if starts[-1]+batch_size < n:
        starts = np.hstack((starts, starts[-1]+batch_size))

    
    for group, shank in zip(np.unique(groups), shanks):

        idx = group == groups
        clu = spike_clusters[idx]
        res = spike_times[idx]

        # Excluding spike times less than n_t on both sides
        try:
            clu = clu[(res>n_t)&(res<(len(fp)-n_t))]
            res = res[(res>n_t)&(res<(len(fp)-n_t))]
        except Exception:
            pass

        while clu.min() < 2:
            clu += 1

        ######################################################################
        # Writing clu
        print(f"Writing clu for shank {shank}")
        clu_file = os.path.join(basepath, basename + '.clu.%i' %(shank))
        n = len(np.unique(clu))
        clu_n = np.hstack(([n], clu))
        np.savetxt(clu_file, clu_n, delimiter = '\n', fmt='%i')


        ######################################################################
        # Writing res
        print(f"Writing res for shank {shank}")
        res_file = os.path.join(basepath, basename + '.res.%i' %(shank))    
        f = open(res_file, "w")
        for t in res:
            f.write(str(t)+"\n")
        f.close()
        
        ######################################################################
        # Writing spk
        print(f"Writing spk for shank {shank}")
        
        idx = np.searchsorted(starts, res)-1
        
        waveforms = np.zeros((len(res),n_t,len(shank_to_channel[group])), dtype=np.int16)
        
        count = 0

        for i, batch_start in tqdm(enumerate(starts), total=len(starts)):

            slice_ = slice(np.maximum(0, batch_start-n_t), np.minimum(len(fp), batch_start+batch_size+n_t))

            lfp = nap.TsdFrame(t=timestep[slice_],d=fp[slice_, shank_to_channel[group]])

            lfp = lfp - np.median(lfp, 0)

            flfp = nap.apply_highpass_filter(lfp, 300, fs)
                        
            for j, spk in enumerate(res[idx==i]):
                t = spk - batch_start + (batch_start-slice_.start)
                waveforms[count] = flfp.values[t-n_t//2:t+n_t//2,:].astype("int16")
                count += 1


        spk_file = os.path.join(basepath, basename + '.spk.%i' %(shank))
        f = open(spk_file, "wb")
        
        for i in tqdm(range(waveforms.shape[0])):            
            for j in range(waveforms.shape[1]):
                for k in range(waveforms.shape[2]):
                    f.write(waveforms[i,j,k])
        f.close()

        ######################################################################
        # Writing fet        
        print(f"Writing fet for shank {shank}")

        features = np.zeros((len(res),len(shank_to_channel[group])*3+3))
        for j in tqdm(range(len(shank_to_channel[group]))):
            tmp = StandardScaler().fit_transform(waveforms[:,:,j])
            features[:,j*3:j*3+3] = PCA(n_components=3).fit_transform(tmp)

        #copied from matlab
        factor = (2**15)/np.max(np.abs(features))
        features *= factor
        features = features.astype(np.int64)


        # Last column is time
        features[:,-1] = res.astype("int64")

        # Second last column is amplitude
        features[:,-2] = (waveforms.max((1,2)) - waveforms.min((1,2))).astype("int64")
        features[:,-2] += (np.min(features[:,-2])+10)

        # Third last column is power of the spikes waveforms
        # features[:,-3] = (np.abs(waveforms).mean((1,2))).astype("int64")
        features[:,-3] = (np.mean(np.power(waveforms, 2), (1,2))/100).astype("int64")
        

        fet_file = os.path.join(basepath, basename + '.fet.%i' %(shank))
        f = open(fet_file, 'w')
        f.writelines(str(features.shape[-1])+'\n')
        for j in tqdm(range(len(features))):            
            tmp = waveforms[j].astype(np.int64)
            f.writelines('\t'.join(features[j].astype('str'))+'\n')
        f.close()





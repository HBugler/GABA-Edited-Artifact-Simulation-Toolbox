# Artifact Simulation Toolbox by Hanna Bugler, Rodrigo Berto, Roberto Souza and Ashley Harris (2023) [last upd. 2024-04-25]
# To generate transients with commonly occuring artifacts from simulated ground truths
# Default function parameter values are written for normalized frequency domain spectra
# Domain Functions: L15-L93, Artifact Functions: L98-L272, Artifact Supporting Functions: : L277-L526

import math
import random
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft, fftshift

########################################################################################################################
# Domain Functions
########################################################################################################################
def to_fids(specs, axis=1):
    '''
    Convert to Fids (time domain)
    :param:     specs: [numSamples, specPoints]
                axis: *provided in case specs axes are swapped*
    :return:    fids: [numSamples, specPoints]
    '''
    return ifft(fftshift(specs, axes=axis), axis=axis)

def to_specs(fids, axis=1):
    '''
    Convert to Specs (frequency domain)
    :param:     fids: [numSamples, specPoints]
                axis: *provided in case fids axes are swapped*
    :return:    specs: [numSamples, specPoints]
    '''
    return fftshift(fft(fids, axis=axis), axes=axis)

def interleave(fids_on, fids_off):
    '''
    Interleave Edited Fids so they appear in order of collection (assumes ON first and a 1,0,1,0,1... interleaving)
    :param:     fidsOn: [numSamples/2, specPoints]
                fidsOff: [numSamples/2, specPoints]
    :return:    fids: [numSamples, specPoints]
    '''
    fids = np.zeros((fids_on.shape[0]*2, fids_on.shape[1]), dtype=complex)
    on, off = 0, 0

    for ii in range(0, fids_on.shape[0]*2):
        if ii%2==0:
            fids[ii, :] = fids_on[on, :]
            on+=1
        else:
            fids[ii, :] = fids_off[off, :]
            off+=1

    return fids

def undo_interleave(fids):
    '''
    Reverses interleaving of edited Fids to obtain both subspectra groups separately (assumes ON first and a 1,0,1,0,1... interleaving)
    :param:     fids: [numSamples, specPoints]
    :return:    fidsOn: [numSamples/2, specPoints]
                fidsOff: [numSamples/2, specPoints]
    '''
    fids_on = np.zeros((int(fids.shape[0]/2), fids.shape[1]), dtype=complex)
    fids_off = np.zeros((int(fids.shape[0]/2), fids.shape[1]), dtype=complex)
    on, off = 0, 0
    
    for ii in range(0, fids.shape[0]):
        if ii%2==0:
            fids_on[on, :] = fids[ii, :]
            on+=1
        else:
            fids_off[off, :] = fids[ii, :]
            off+=1

    return fids_on, fids_off

def scale(fids):
    '''
    Scale data by the max of the absolute value of the complex data to apply artifact functions
    :param:     fids: [numSamples, specPoints]
    :return:    fids: [numSamples, specPoints] **normalized**
                scaleFact(scale factor): integer
    '''
    scale_fact = np.max(abs(to_specs(fids)))
    norm_specs = to_specs(fids)/scale_fact
    return to_fids(norm_specs), scale_fact

def undo_scale(fids, scale_fact):
    '''
    Undo scaling
    :param:     fids: [numSamples, specPoints]
                scaleFact(scale factor): integer
    :return:    fids: [numSamples, specPoints] **normalized**
    '''
    norm_specs = to_specs(fids)*scale_fact
    return to_fids(norm_specs)

########################################################################################################################
# Artifact functions
########################################################################################################################
def add_time_domain_noise(fids, noise_level=10):
    '''
    Add complex time domain noise
    :param:     fids: [numSamples, specPoints]
                noiseLev(standard deviation of noise level): integer 
    :return:    fids: [numSamples, specPoints]  ** with Gaussian White Noise**
    '''
    fids = (fids.real + np.random.normal(0, noise_level, size=(fids.shape))) + \
           (fids.imag + np.random.normal(0, noise_level, size=(fids.shape))) * 1j
    
    return fids

def add_progressive_motion_artifact(time, fids, off_var=None, slope_var=None, start_trans=None, num_affected_trans=None):
    '''
    To add a frequency drift mimicking a participant's head moving in one direction over time
    :param:     time: [specPoints]
                fids: [numSamples, specPoints]
                freq_offset_var(variance at each step within the drift): integer
                freq_shift(overall frequency shift to accomplish between first and last transient): integer
                startTrans(number of transient to start at): integer
                numTrans(number of transients affected): integer
    :return:    fids: [numSamples, specPoints]  ** with random frequency shift**
                [startTrans, numTrans]: 2-element list 
    '''
    fids, trans_affected = add_freq_shift_linear(time, fids, freq_offset_var=off_var, freq_shift=slope_var, start_trans=start_trans, num_trans=num_affected_trans)

    return fids, trans_affected

def add_subtle_motion_artifact(fids, time, freq_shift_var=None, phase_shift_var=None, cluster=False, num_affected_trans=None):
    '''
    To add a small frequency and phase shifts to certain transients to mimic some participant motion
    :param:     fids:[numSamples, specPoints]
                time: [specPoints]
                freqShiftVar(+/- range of frequency shifts): integer/float
                phaseShiftVar(+/- range of phase shifts): integer/float
                cluster(True means affected transients will be consecutive): boolean
                numAffectedTrans(default is ALL transients): integer
    :return:    fids:[numSamples, specPoints]**with subtle motion(s) artifact inserted**
    '''   
    fids, num_trans_fs = add_freq_shift_random(time, fids, freq_var=freq_shift_var, cluster=cluster, num_trans=num_affected_trans)
    fids, num_trans_ps = add_phase_shift_random(fids, phase_var=phase_shift_var, cluster=cluster, num_trans=num_affected_trans)

    return fids, [num_trans_fs, num_trans_ps]

def add_disruptive_motion_artifact(fids, time, ppm, amp=None, lvs=None, cluster=False, mot_locs=None, nmb_motion=1):
    '''
    To add a linebroadening and baseline changes to mimic large participant motion
    :param:     fids:[numSamples, specPoints]
                time: [specPoints]
                ppm: [specPoints]
                amp(amplitude): integer/float
                lvs(lineshape variance): integer/float
                cluster(True means affected transients will be consecutive): boolean
                nmb_motion: integer
    :return:    fids:[numSamples, specPoints]**with disruptive motion(s) artifact inserted**
                motLocs: [*list of integers where disruptive motion(s) have been added based on length nmb_motion*]
    '''
    fids, mot_locs = motion_linebroad_artifact(fids, time, amp=amp, lvs=lvs, cluster=cluster, mot_locs=mot_locs, nmb_motion=nmb_motion)   
    fids = motion_baseline_artifact(fids, ppm, mot_locs=mot_locs)

    return fids, mot_locs

def add_lipid_artifact(fids, ppm, edited=False, glob_loc=None, avg_amp=None, cluster=False, lp_locs=None, nmb_lps=1):
    '''
    To add lipid artifacts to a select number of transients
    :param:     fids:[numSamples, specPoints]
                ppm: [specPoints]
                glob_loc(global location for lipid peak): integer/float
                avgAmp(average amplitude of the lipid peak): integer
                cluster(True means affected transients will be consecutive): boolean
                nmb_lps(default is ALL): integer
    :return:    fids:[numSamples, specPoints]**with lipid artifact(s) inserted**
                lpLocs: [*list of integers where lipid artifact(s) have been added based on length nmb_lps*]
    '''
    fids, lp_locs, glob_loc_final = lipid_peak_contamination(fids, ppm, edited=edited, avg_amp=avg_amp, glob_loc=glob_loc, cluster=cluster, lp_locs=lp_locs, nmb_lp=nmb_lps)
    fids = lipid_baseline_contamination(fids, ppm, glob_loc=glob_loc_final, lp_locs=lp_locs)

    return fids, lp_locs

def add_ghost_artifact(fids, time, amp=None, cs=None, phase=None, tstart=None, tfinish=None, cluster=False, gs_locs=None, nmb_ghosts=1):  
    '''
    To add a ghost artifact to a select number of transients (Based on Berrington et al. 2021)
    :param:     fids: [numSamples, specPoints]
                time: [specPoints]
                amp (amplitude): [*list of length nmb_ghosts*] 
                cs (chemical shift in ppm): [*list of length nmb_ghosts*] 
                phase (phase of artifact in radians): [*list of length nmb_ghosts*] 
                tstart (start time (ms) of artifact in time domain): [*list of length nmb_ghosts*] 
                tfinish (end time (ms) of artifact in time domain): [*list of length nmb_ghosts*] 
                cluster(whether transients affected are consecutive): boolean
                gLocs (ghost locations/transient number): [*list of length nmb_ghosts*]
                nmb_ghosts (number of ghost artifacts in specific scan): integer
    :return:    fids: [numSamples, specPoints]  **with ghost(s) artifact inserted**
                gLocs (ghost locations/transient number): [*list of length nmb_ghosts in order*]
    '''
    func_def = []
    lf, t_all = (2*np.pi*127), np.max(time, axis=0)

    # select random FIDs/Transients to insert ghost artifact
    if gs_locs is None or len(gs_locs)!=nmb_ghosts:
        if cluster is False:
            gs_locs = np.random.choice(range(0, fids.shape[0]), size=nmb_ghosts, replace=False)
        else:
            start = int(np.random.uniform(0, ((fids.shape[0]/2)-nmb_ghosts), size=1))
            gs_locs = range(start, start+nmb_ghosts)
        func_def.append('Locations')

    # select random location in FID (time domain) for artifact to occur (start at 0.35 (700) - 0.5 (1200) ms, end at 0.6 (800) - 0.75 (1700) ms)
    if tstart is None or len(tstart)!=nmb_ghosts:
        tstart = np.random.uniform(700, 1000, size=nmb_ghosts)
        func_def.append('Start Times')

    tstart = np.array(tstart)
    if (tfinish is None) or (len(tfinish)!=nmb_ghosts) or (any(tstart >= tfinish)):
        tfinish =tstart + np.random.uniform(100, 500, size = nmb_ghosts)
        func_def.append('End Times')    

    # amplitude is also dependent on length of time of ghost
    tfinish = np.array(tfinish)
    if phase is None or len(phase)!=nmb_ghosts:
        phase = np.random.uniform(0.1, 1.9, size=nmb_ghosts)*math.pi
        func_def.append('Phases')

    if amp is None or len(amp)!=nmb_ghosts:
        amp = np.random.uniform(10, 100, size=nmb_ghosts)
        func_def.append('Amplitudes')

    if cs is None or len(cs)!=nmb_ghosts:
        cs = np.random.uniform(-1.0, 5.0, size=nmb_ghosts)
        func_def.append('Chemical Shifts')
    # if lf is kept the same, remap cs values user value (equivalent spectrum value at current lf) [5(1), 4(2), 3(3), 2(4), 1(5), 0(6), -1(7)]
    else:
        cs = [6 - x  for x in cs]
    
    gs_locs = np.array(gs_locs)
    phase = np.array(phase)
    amp = np.array(amp)
    cs = np.array(cs)

    # insert ghost artifact into FID(s)
    for ii in range(0, nmb_ghosts):
        s, f = int(tstart[ii]),int(tfinish[ii])
        fids[gs_locs[ii], s:f] = fids[gs_locs[ii], s:f]*amp[ii]*np.exp(-abs(time[s:f])/t_all)*np.exp(-1j*(lf*(1-cs[ii])*time[s:f]+phase[ii]))
        
    if func_def:
        print(f'Non-user defined parameters: {func_def}')

    return fids, np.sort(gs_locs)


def add_eddy_current_artifact(fids, time, amp=None, tc=None, cluster=False, ec_locs=None, nmb_ecs=1):
    '''
    To add an Eddy Current artifact into specified number of transient (adapted from Eddy Current Artifact in FID-A (Simpson et al. 2017)
    :param:     fids:[numSamples, specPoints]
                time: [specPoints]
                amp(amplitude): [*list of length nmb_ec*] 
                tc(time constant): [*list of length nmb_ec*]
                cluster(whether transients affected are consecutive): boolean
                EcLocs(eddy current locations/transient number): [*list of length nmb_ec*]
                nmb_ec(number of eddy current artifacts in specific scan): integer
    :return:    fids:[numSamples, specPoints]**with EC artifact(s) inserted**, 
                EcLocs(EC locations/transient number):[*list of length nmb_ec*]
    '''
    func_def = []

    # select random FIDs/Transients to insert EC artifact
    if ec_locs is None or len(ec_locs)!=nmb_ecs:
        if cluster is False:
            ec_locs = np.random.choice(range(0, fids.shape[0]), size=nmb_ecs, replace=False)
        else:
            start = int(np.random.uniform(0, ((fids.shape[0]/2)-nmb_ecs), size=1))
            ec_locs = np.array(range(start, start+nmb_ecs))
        func_def.append('Locations')

    # variables related to the artifact: amplitude (Hz) and time constant of decaying artifact (in seconds)
    if amp is None or len(amp)!=nmb_ecs:
        amp = np.random.uniform(0.500, 5.000, size=nmb_ecs)
        func_def.append('Amplitudes')

    amp = np.array(amp)[:, np.newaxis].repeat(time.shape[0], axis=1)
    if tc is None or len(tc)!=nmb_ecs:
        tc = np.random.uniform(0.250, 1.000, size=nmb_ecs)
        func_def.append('Time Constants')

    tc = np.array(tc)[:, np.newaxis].repeat(time.shape[0], axis=1)

    
    # insert EC artifact into FID
    time = time[np.newaxis, :].repeat(nmb_ecs, axis=0)
    fids[ec_locs, :] = fids[ec_locs, :] * (np.exp(-1j * time * (amp * np.exp(time / tc)) * 2 * math.pi))
    
    if func_def:
        print(f'Non-user defined parameters: {func_def}')
    
    return fids, np.sort(ec_locs)

########################################################################################################################
# Artifact Supporting functions
########################################################################################################################
def motion_linebroad_artifact(fids, time, amp=None, lvs=None, cluster=False, mot_locs=None, nmb_motion=1):
    '''
    Add line broadening artifact (mimicking large motion)
    :param:     fids: [numSamples, specPoints]
                time: [specPoints]
                cluster(whether transients affected are consecutive): boolean
                motLocs(locations (transient numbers) where motion artifact is to occur in ppm): [*list of length nmb_lp*]
                nmb_motion: integer (number of transients affected)
    :return:    fids: [numSamples, specPoints]  ** with linebroadening artifact**
                motLocs: list of integers (which transients are affected)
    '''
    func_def = []
    # select random FIDs/Transients to insert EC artifact
    if mot_locs is None or len(mot_locs)!=nmb_motion:
        if cluster is False:
            mot_locs = np.random.choice(range(0, fids.shape[0]), size=nmb_motion, replace=False)
        else:
            start = int(np.random.uniform(0, ((fids.shape[0]/2)-nmb_motion), size=1))
            mot_locs = np.array(range(start, start+nmb_motion))
        func_def.append('Locations')

    # variables related to the artifact: amplitude and lineshape variance
    if amp is None or len(amp)!=nmb_motion:
        amp = np.random.uniform(1.500, 2.250, size=nmb_motion)
        func_def.append('Amplitudes')

    if lvs is None or len(lvs)!=nmb_motion:
        lvs = np.random.uniform(20, 40, size=nmb_motion)
        func_def.append('Lineshape Variance')

    amp = np.array(amp)[:, np.newaxis].repeat(time.shape[0], axis=1)
    lvs = np.array(lvs)[:, np.newaxis].repeat(time.shape[0], axis=1)

    # insert LB artifact into FID
    time = time[np.newaxis, :].repeat(nmb_motion, axis=0)
    fids[mot_locs, :] = fids[mot_locs, :] * (amp * np.exp(-time * lvs))
    
    if func_def:
        print(f'Non-user defined parameters: {func_def}')

    return fids, np.sort(mot_locs)


def motion_baseline_artifact(fids, ppm, mot_locs):
    '''
    Add change in baseline (mimicking large motion)
    :param:     fids: [numSamples, specPoints]
                ppm: [specPoints]
                motLocs(locations (transient numbers) where motion artifact is to occur in ppm): [*list of length nmb_lp*]
    :return:    fids: [numSamples, specPoints]  ** with motion baseline artifact**
    '''
    # The following attributes need to be defined: self.motion_locations
    # get specs
    specs = to_specs(fids)
    num_bases = []
    
    # get the number of sine baselines per motion corrupted transient
    for i in range(len(mot_locs)):
        p = random.randint(1, 4)
        num_bases.append(p)
    # generate the motion corrupted baseline and insert into transient
    dc_shift = np.argmax(ppm <= -1.0)
    for i in range(0, len(mot_locs)):
        amp, cyc = np.random.uniform(0.05, 0.10, size=num_bases[i]), np.random.uniform(0.1, 3, size=num_bases[i])
        for k in range(0, num_bases[i]):
            frame_start_ppm = np.argmax(ppm <= np.random.uniform(ppm[0], ppm[1024]))
            frame_end_ppm = np.argmax(ppm <= np.random.uniform(ppm[1024], ppm[-1]))
            # creation of baseline contamination
            base = np.zeros(len(ppm))
            sine = abs(amp[k] * np.sin(cyc[k] * ppm[frame_start_ppm:frame_end_ppm]))
            base[frame_start_ppm:frame_end_ppm] = signal.savgol_filter(sine, 201, 5, mode='nearest')
            specs[mot_locs[i], :] = base + specs[mot_locs[i], :] - specs[mot_locs[i], dc_shift]

    return to_fids(specs)

def lipid_peak_contamination(fids, ppm, edited=False, glob_loc=None, avg_amp=None, cluster=False, lp_locs=None, nmb_lp=1):
    '''
    Add lipid peak artifact
    :param:     fids: [numSamples, specPoints]
                ppm: [specPoints]
                edited(whether transients are part of an editing scheme with interleaving (1-0-1-0...)): boolean
                glob_loc(overall location where lipid peak/artifact is to occur in ppm): [*list of length nmb_lp*]
                avgAmp(average amplitudeo of the lipid peak): integer
                cluster(whether transients affected are consecutive): boolean
                lpLocs(locations (transient numbers) where lipid peak/artifact is to occur in ppm): [*list of length nmb_lp*]
                nmb_lp: list of integers (which transients are affected)
    :return:    fids: [numSamples, specPoints]  ** with lipid peak artifact**
                lpLocs: list of integers (which transients are affected)
    '''
    # get specs and location of lipid contaminations
    specs = to_specs(fids)
    func_def = []
    
    # select random FIDs/Transients to insert EC artifact
    if lp_locs is None or len(lp_locs)!=nmb_lp:
        if cluster is False:
            lp_locs = np.sort(np.random.choice(range(0, fids.shape[0]), size=nmb_lp, replace=False))
        else:
            start = int(np.random.uniform(0, ((fids.shape[0]/2)-nmb_lp), size=1)) 
            lp_locs = np.array(range(start, start+nmb_lp))
        func_def.append('Locations')
    
    if glob_loc is None or len(glob_loc)!=nmb_lp:
        glob_loc = np.random.uniform(1.525, 1.575, size=nmb_lp)
        func_def.append('Global Locations')

    if avg_amp is None:
        avg_amp = 0.0005
        func_def.append('Maximum Amplitude')

    amp1, amp2 = np.random.uniform(avg_amp*0.75, avg_amp*1.25, size=nmb_lp), np.random.uniform(avg_amp*0.75, avg_amp*1.25, size=nmb_lp)
    loc1, loc2 = np.random.uniform(glob_loc-0.005, glob_loc+0.05, size=nmb_lp), np.random.uniform(glob_loc-0.005, glob_loc+0.05, size=nmb_lp)
    sig1, sig2 = np.random.uniform(0.01, 0.10, size=nmb_lp), np.random.uniform(0.01, 0.10, size=nmb_lp)

    ppm = ppm[np.newaxis, :].repeat(nmb_lp, axis=0)
    gauss_p1, gauss_p2 = np.zeros((nmb_lp, ppm.shape[1])),  np.zeros((nmb_lp, ppm.shape[1]))
    for ii in range (0, nmb_lp):
        if edited is True and lp_locs[ii]%2==0:
            rnd = np.random.uniform(0.001, 0.005)
            gauss_p1[ii, :] = (amp1[ii]+rnd) * (np.exp(-np.power(ppm[ii, :] - loc1[ii], 2.) / (2 * np.power(sig1[ii], 2.))))
            gauss_p2[ii, :] = (amp2[ii]+rnd) * (np.exp(-np.power(ppm[ii, :] - loc2[ii], 2.) / (2 * np.power(sig2[ii], 2.))))
        else:
            gauss_p1[ii, :] = amp1[ii] * (np.exp(-np.power(ppm[ii, :] - loc1[ii], 2.) / (2 * np.power(sig1[ii], 2.))))
            gauss_p2[ii, :] = amp2[ii] * (np.exp(-np.power(ppm[ii, :] - loc2[ii], 2.) / (2 * np.power(sig2[ii], 2.))))
    gauss_final = gauss_p1 + gauss_p2
    specs[lp_locs, :] = specs[lp_locs, :] + gauss_final

    if func_def:
        print(f'Non-user defined parameters: {func_def}')

    return to_fids(specs), lp_locs, glob_loc


def lipid_baseline_contamination(fids, ppm, glob_loc, lp_locs=None):
    '''
    Add lipid baseline artifact
    :param:     fids: [numSamples, specPoints]
                ppm: [specPoints]
                lpLocs(which transients are affected): list of integers
    :return:    fids: [numSamples, specPoints]  ** with lipid baseline artifact**
    '''
    # get specs and randomize baseline contamination
    specs = to_specs(fids)

    if len(lp_locs)<1:
        print('Lipid Locations need to be provided')
    else:
        glob_loc = np.mean(glob_loc)
        amp_min, amp_maj = np.random.uniform(0.0005, 0.0010, size=len(lp_locs)), np.random.uniform(0.0010, 0.0020, size=len(lp_locs))
        cyc_min, cyc_maj = np.random.uniform(0.090, 0.095, size=len(lp_locs)), np.random.uniform(0.120, 0.125, size=len(lp_locs))
        # creation of baseline contamination
        frame_000ppm, frame_198ppm = np.where(ppm <= glob_loc-1.50)[0][-1], np.where(ppm <= glob_loc-0.02)[0][-1]
        frame_200ppm, frame_303ppm = np.where(ppm <= glob_loc+0.02)[0][-1], np.where(ppm <= glob_loc+1.50)[0][-1]

        sine_maj, sine_min = np.zeros((len(lp_locs),len(ppm))), np.zeros((len(lp_locs),len(ppm)))
        ppm = ppm[np.newaxis, :].repeat(len(lp_locs), axis=0)
        for ii in range(0, len(lp_locs)):
            sine_maj[ii, frame_000ppm:frame_198ppm] = abs(amp_maj[ii] * np.sin((cyc_maj[ii] * ppm[ii, frame_000ppm:frame_198ppm])))
            sine_min[ii, frame_200ppm:frame_303ppm] = abs(amp_min[ii] * np.cos((cyc_min[ii] * ppm[ii, frame_200ppm:frame_303ppm])))
        specs[lp_locs, :] = sine_min + sine_maj + specs[lp_locs, :]

    return to_fids(specs)


def add_freq_shift_random(time, fids, freq_var=None, cluster=False, num_trans=None):
    '''
    Add random frequency shifts
    :param:     time: [specPoints]
                fids: [numSamples, specPoints]
                freq_var(+/- range of frequency shifts): integer
                cluster(whether transients affected are consecutive): boolean
                numTrans(number of transients affected): integer
    :return:    fids: [numSamples, specPoints]  ** with random frequency shift**
                numTrans(number of transients affected): integer 
    '''
    func_def = []

    # select how many transients will be affected, clustering effect and range of shifts
    if num_trans is None:
        num_trans= fids.shape[0]
        func_def.append('Number of Affected Transients')

    if cluster is True:
        start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1))
        rnd = np.array(range(start, start+num_trans))
    else:
        rnd = np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False)
        func_def.append('Clustering')

    if freq_var is None:
        freq_var = np.random.uniform(2, 20, size=1)
        func_def.append('Frequency Shift Variance')

    f_noise = np.random.uniform(low=-abs(freq_var), high=freq_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)
    
    # insert phase shifts into scan
    time = time[np.newaxis, :].repeat(num_trans, axis=0)
    fids[rnd, :] = fids[rnd, :] * np.exp(-1j * time * f_noise * 2 * math.pi)

    if func_def:
        print(f'Non-user defined parameters: {func_def}')

    return fids, np.sort(rnd)


def add_freq_shift_linear(time, fids, freq_offset_var=None, freq_shift=None, start_trans=None, num_trans=None):
    '''
    Add linear frequency shift (drift)
    :param:     time: [specPoints]
                fids: [numSamples, specPoints]
                freq_offset_var(variance at each step within the drift): integer
                freq_shift(overall frequency shift to accomplish between first and last transient): integer
                startTrans(number of transient to start at): integer
                numTrans(number of transients affected): integer
    :return:    fids: [numSamples, specPoints]  ** with random frequency shift**
                [startTrans, numTrans]: 2-element list 
    '''
    func_def = []
    if num_trans is None:
        num_trans = int(np.random.uniform(2, (fids.shape[0]-1)))
        func_def.append('Number of Transients Affected')

    if start_trans is None:
        start_trans = int(np.random.uniform(0, (fids.shape[0]-num_trans)))
        func_def.append('First Transient')

    if freq_offset_var is None:
        freq_offset_var = np.random.uniform(0.01, 2)
        func_def.append('Offset Variation')

    if freq_shift is None:
        freq_shift = np.random.uniform(0.05*num_trans, 0.15*num_trans)
        func_def.append('Overall Frequency Shift')

    end_trans = start_trans+num_trans
    slope = np.arange(0, freq_shift, freq_shift/num_trans)
    noise = np.random.normal(0, freq_offset_var, size=num_trans) + slope
    noise = noise[:, np.newaxis].repeat(2048, axis=1)
    time = time[np.newaxis, :].repeat(num_trans, axis=0)
    fids[start_trans:end_trans, :] = fids[start_trans:end_trans, :]*np.exp(-1j*time*noise*2*math.pi)
    
    if func_def:
        print(f'Non-user defined parameters: {func_def}')
    
    return fids, [start_trans, num_trans]


def add_phase_shift_random(fids, phase_var=None, cluster=False, num_trans=None):
    '''
    Add random phase shifts
    :param:     fids: [numSamples, specPoints]
                phase_var(+/- range of phase shifts): integer 
                cluster(whether transients affected are consecutive): boolean
                numTrans(number of transients affected): integer
    :return:    fids: [numSamples, specPoints]  ** with random phase shifts**
                numTrans(number of transients affected): integer
    '''
    func_def = []

    # select how many transients will be affected, clustering effect and range of shifts
    if num_trans is None:
        num_trans= fids.shape[0]
        func_def.append('Number of Affected Transients')

    if cluster is True:
        start = int(np.random.uniform(0, ((fids.shape[0]/2)-num_trans), size=1))
        rnd = np.array(range(start, start+num_trans))

    else:
        rnd = np.random.choice(range(0, fids.shape[0]), size=num_trans, replace=False)
        func_def.append('Clustering')

    if phase_var is None:
        phase_var = np.random.uniform(5, 90, size=1)
        func_def.append('Phase Shift Variance')

    p_noise = np.random.uniform(low=-abs(phase_var), high=phase_var, size=(num_trans, 1)).repeat(fids.shape[1], axis=1)

    # insert phase shifts into scan
    fids[rnd, :] = fids[rnd, :] * np.exp(-1j * p_noise * math.pi / 180)

    if func_def:
        print(f'Non-user defined parameters: {func_def}')
    
    return fids, np.sort(rnd)

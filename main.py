# Artifact Simulation Toolbox by Hanna Bugler, Rodrigo Berto, Roberto Souza and Ashley Harris (2023)
# For the simulation of Eddy Currents, Spurious Echoes, Lipid Contamination and Motion Corruption

import numpy as np
from random import randint
from transient_maker import TransientMaker
from Display import display_artifact, display_transToDate, saveData

#######################################################################################################################
# USER VARIABLES
#######################################################################################################################
# Expected format of FIDS is [number of GTs, number of spectral points] as .npy file
totalGroundTruths = 100                                                       # Number of Ground Truth FIDs
transient_count = 160                                                         # Number of Difference Transients in Scan

# Location (Directory) Containing the Following Information
ON_fid_gt_location = "C:/Users/Hanna B/Desktop/Research/Thesis_Project/SpecArtifactGen/inputs/fidsON_QA_Test_2048_GTs.npy"
OFF_fid_gt_location = "C:/Users/Hanna B/Desktop/Research/Thesis_Project/SpecArtifactGen/inputs/fidsOFF_QA_Test_2048_GTs.npy"
ppm_location = "C:/Users/Hanna B/Desktop/Research/Thesis_Project/SpecArtifactGen/inputs/on_ppm_2048.csv"
time_location = "C:/Users/Hanna B/Desktop/Research/Thesis_Project/SpecArtifactGen/inputs/on_t_2048.csv"

#######################################################################################################################
# CREATION OF 'NORMAL' TRANSIENTS FROM BASE FID
#######################################################################################################################
all_specs, all_fids = [], []

for groundTruthNumber in range(0, totalGroundTruths):                         # Going through each ground truths

    # Number of Corrupt Transients to Substitute into THIS Scan
    transient_corr_count = randint(int(transient_count/8), int(transient_count/8))

    # CREATING A GROUND TRUTH SCAN OBJECT
    normalTransients = TransientMaker(groundTruthNumber, ON_fid_gt_location, OFF_fid_gt_location, ppm_location, time_location, transient_count=transient_count)
    on_gt_fid, off_gt_fid = normalTransients.get_fids()
    diff_gt_fid = normalTransients.get_differenceFids()[0, :, 0]
    diff_gt_spec = normalTransients.get_differenceSpecs()[0, :, 0]

    #######################################################################################################################
    # CREATION OF NORMAL (BASELINE) TRANSIENTS
    #######################################################################################################################

    # NO INTENTIONAL ARTIFACTS (ADD RANDOM CHANGE TO NOISE, FREQUENCY AND PHASE OFFSETS FROM ORIGINAL TRANSIENTS)
    # VALUES ARE SELECTED TO MIMIC SNR OF ~25 AS PER THE DATA PROVIDED BY BIG GABA AND QUOTED IN Mikkelsen et al. 2018 J. of Neuro. Methods
    normNoise, normFreqVar, normPhaseVar = np.random.uniform(0.10, 0.25, size=1), np.random.uniform(2, 5, size=1), np.random.uniform(5, 10, size=1)

    # TIME DOMAIN AMPLITUDE NOISE (NORMAL DISTRIBUTION)
    normalTransients.add_time_domain_noise(noise_level=normNoise)
    # RANDOM FREQUENCY SHIFT (NORMAL DISTRIBUTION - STANDARD DEVIATION IN HZ)
    normalTransients.add_freq_shift_random(freq_var=normFreqVar)
    # RANDOM PHASE SHIFT (NORMAL DISTRIBUTION - STANDARD DEVIATION IN DEGREES)
    normalTransients.add_phase_shift_random(phase_var=normPhaseVar)

    # GET TYPICAL SPECS AND FIDS
    on_normal_specs, off_normal_specs = normalTransients.get_specs()
    diff_normal_specs = normalTransients.get_differenceSpecs()
    on_normal_fids, off_normal_fids = normalTransients.get_fids()
    diff_normal_fids = normalTransients.get_differenceFids()

    #######################################################################################################################
    # CREATION OF CORRUPT TRANSIENTS
    #######################################################################################################################
    corruptTransients = TransientMaker(groundTruthNumber, ON_fid_gt_location, OFF_fid_gt_location, ppm_location, time_location, transient_count=transient_corr_count)

    # TIME DOMAIN SPURIOUS ECHOES (FD GHOSTS) ARTIFACTS
    # nmb_ghost: number of echoes in scan, start and stop time in ms (of echo), amp: amplitude of echo, mult: phase multiplier in rads, cs: chemical shift in ppm
    corruptTransients.add_ghost_artifact(nmb_ghost=None, start=None, stop=None, amp=None, mult=None, cs=None)

    # TIME DOMAIN EDDY CURRENT ARTIFACTS
    # nmb_ec: number of eddy currents in scan, amp: amplitude of EC, tc: time constant of decaying artifact (in seconds)
    corruptTransients.add_EddyCurrent_artifact(nmb_ec=None, amp=None, tc=None)

    # LARGE ONE-TIME MOTION
    # TIME DOMAIN LINE BROADENING and FREQUENCY DOMAIN BASELINE MOTION ARTIFACT
    # nmb_motion: number of large motion artifacts in scan, amp: amplitude of line broadening, lvs: lineshape variance of artifact
    corruptTransients.motion_lineBroad_artifact(nmb_motion=None, amp=None, lvs=None)
    corruptTransients.motion_baseline_artifact()

    # FREQUENCY DOMAIN LIPID PEAK AND BASELINE ARTIFACT
    # nmb_lp: number of lipid contaminants in scan
    corruptTransients.lipid_peak_contamination(nmb_lp=None)
    corruptTransients.lipid_baseline_contamination()

    # ADD RANDOM CHANGE TO NOISE, FREQUENCY AND PHASE OFFSETS TO BASELINE TRANSIENTS
    randNoiseChange, randFreqChange, randPhaseChange = np.random.uniform(3, 5, size=1), np.random.uniform(3, 5, size=1), np.random.uniform(3, 5, size=1)
    randFreqSlope = np.random.uniform(10, 40, size=1)

    # SMALL RANDOM MOTION
    # RANDOM FREQUENCY SHIFT (NORMAL DISTRIBUTION - STANDARD DEVIATION IN HZ)
    corruptTransients.add_freq_shift_random(freq_var=normFreqVar * randFreqChange)
    # RANDOM PHASE SHIFT (NORMAL DISTRIBUTION - STANDARD DEVIATION IN DEGREES)
    corruptTransients.add_phase_shift_random(phase_var=normPhaseVar + 2 * randPhaseChange)

    # SMALL PROGRESSIVE MOTION
    # LINEAR FREQUENCY SHIFT (NORMAL DISTRIBUTION - STANDARD DEVIATION IN HZ)
    corruptTransients.add_freq_shift_linear(freq_offset_var=0, freq_slope_var=randFreqSlope)

    # TIME DOMAIN AMPLITUDE NOISE (NORMAL DISTRIBUTION)
    corruptTransients.add_time_domain_noise(noise_level=normNoise*randNoiseChange)

    # GET CORRUPT SPECS AND FIDS
    on_corr_specs, off_corr_specs = corruptTransients.get_specs()
    diff_corr_specs = corruptTransients.get_differenceSpecs()
    on_corr_fids, off_corr_fids = corruptTransients.get_fids()
    diff_corr_fids = corruptTransients.get_differenceFids()

    # DISPLAY EXAMPLES
    # display_artifact(diff_gt_fid, diff_gt_spec, corruptTransients, artifact_name='Lipid Artifact', location=corruptTransients.LP_locations)
    # display_transToDate(corruptTransients, gt=diff_gt_spec)

    #######################################################################################################################
    # CONFIGURE WHOLE SCAN (NORMAL + CORRUPT TRANSIENTS)
    #######################################################################################################################
    # INSERT CORRUPT TRANSIENTS AT RANDOM LOCATIONS WITHIN SCAN
    on_all_specs, off_all_specs = normalTransients.insert_corrupt(corruptTransients)
    diff_all_specs = normalTransients.get_differenceSpecs()
    on_all_fids, off_all_fids = normalTransients.get_fids()
    diff_all_fids = normalTransients.get_differenceFids()

    # GET LOCATIONS OF CORRUPT TRANSIENTS
    normalTransients.nonArtifTrans = np.copy(diff_all_specs)
    normalTransients.nonArtifTrans = (np.delete(normalTransients.nonArtifTrans, normalTransients.artifact_locations, axis=2))

    ########################################################################################################################
    # METRIC CALCULATIONS FOR SNR, LINEWIDTH AND SHAPE SCORE
    ########################################################################################################################
    # GET METRICS FOR ALL (NORMAL + CORRUPT) TRANSIENTS
    normalTransients.GABA_snr = normalTransients.get_SNR(diff_normal_specs, single='no', metab='GABA')
    normalTransients.ss = normalTransients.get_ShapeScore(diff_normal_specs, diff_gt_spec, single='no')
    normalTransients.GABA_lw = normalTransients.get_LW(diff_normal_specs, single='no', metab='GABA')

    #######################################################################################################################
    # Print Scan Results and Save Values
    #######################################################################################################################
    normalTransients.print_results(corrupt='no', metrics='yes')
    all_specs.append(diff_all_specs)
    all_fids.append(diff_all_fids)

# saveData(all_specs, all_fids)
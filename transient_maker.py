# Artifact Simulation Toolbox by Hanna Bugler, Rodrigo Berto, Roberto Souza and Ashley Harris (2023)
# To simulate the artifacts, get metric values and apply domain specific functions

import numpy as np
import math
import random
from scipy import signal


class TransientMaker:

    def __init__(self, groundTruthNum, ON_loc, OFF_loc, ppm_loc, time_loc, transient_count):
        # load data
        base_on_fids = np.load(ON_loc)
        base_off_fids = np.load(OFF_loc)
        ppm = np.loadtxt(ppm_loc, delimiter=",")[:, 0].flatten()
        t = np.loadtxt(time_loc, delimiter=",")[:, 0].flatten()


        # ATTRIBUTES
        # for holding transients
        self.base_fids_on = base_on_fids[groundTruthNum].reshape(1, -1).copy()
        self.base_fids_off = base_off_fids[groundTruthNum].reshape(1, -1).copy()
        assert self.base_fids_on.shape[0] == self.base_fids_off.shape[0]
        self.sample_count = self.base_fids_on.shape[0] + self.base_fids_on.shape[0]
        self.edit_on = np.tile(np.array([1, 0]), int(self.sample_count / 2))
        self.fids = self.base_fids_on.repeat(2, axis=0) * self.edit_on.reshape(-1, 1) + self.base_fids_off.repeat(2, axis=0) * (1 - self.edit_on.reshape(-1, 1))
        self.fids = self.fids.reshape(self.fids.shape[0], self.fids.shape[1], 1).repeat(transient_count, axis=2)

        self.base_specs_on = to_specs(self.base_fids_on)
        self.base_specs_off = to_specs(self.base_fids_off)
        self.corrupt_specs = []
        self.good_specs = []

        # For Basic Transient Information
        self.t, self.ppm = t, ppm
        self.transient_count = transient_count
        self.groundTruthNum = groundTruthNum

        # For Artifact Information
        self.nonArtifTrans = []
        self.artifact_locations = None
        self.locationsLeft = list(range(0, transient_count - 1))
        self.ghost_locations, self.EC_locations, self.motion_locations, self.LP_locations = [], [], [], []

        # For Metrics
        self.GABA_snr, self.NAA_snr = None, None
        self.GABA_lw, self.NAA_lw = None, None
        self.ss = None

    ########################################################################################################################
    # Output summary function
    ########################################################################################################################
    def print_results(self, corrupt=None, metrics=None):
        '''
        Prints select results of the simulations.
        Most class attributes need to be defined for function to work.

        :param corrupt: binary ('yes' or 'no') that allows you to display corruption results.
        :param metrics: binary ('yes' or 'no') that allows you to display metric results.
        :return: None
        '''

        print(f"")
        print(f"---------------------------------------------------------")
        print(f"          Results Summary for Ground Truth (Scan) #{self.groundTruthNum + 1}")
        print(f"---------------------------------------------------------")
        if corrupt == 'yes':
            print(f"Number of Artifact/Corrupt Transients: {self.artifact_locations.shape[0]}")
            print(f"Location (transient #) of Corrupt Transients: {self.artifact_locations}")
            print(f"Location (transient #) of Ghost Artifacts: {self.ghost_locations}")
            print(f"Location (transient #) of Eddy Current Artifacts: {self.EC_locations}")
            print(f"Location (transient #) of Motion Artifacts: {self.motion_locations}")
            print(f"Location (transient #) of Lipid Artifacts: {self.LP_locations}")
        if metrics == 'yes':
            print(f"GABA SNR of Mean Spectrum: {np.round(self.GABA_snr, 3)}")
            print(f"GABA LW of Mean Spectrum: {np.round(self.GABA_lw, 3)} Hz")
            print(f"GABA LW of Ground Truth Spectrum is {np.round(self.get_LW((self.base_specs_on - self.base_specs_off), single='yes', metab='GABA'), 3)} Hz")
            print(f"Shape Score of GABA and GLX: {np.round(self.ss, 3)}/1.000")
        print(f"---------------------------------------------------------")

    ########################################################################################################################
    # Typical noise functions
    ########################################################################################################################
    def add_time_domain_noise(self, noise_level=10):
        '''
        Adds random Gaussian amplitude noise to the FID signal.

        :param noise_level: standard deviation of the random amplitude noise.
        :return: None
        '''

        noise = np.random.normal(0, noise_level, size=self.fids.shape)
        self.fids = self.fids + noise


    def add_freq_shift_random(self, freq_var=5):
        '''
        Adds random frequency shifts (with normal distribution) to the FID signal.

        :param freq_var: standard deviation of the frequency shifts distribution of the signal.
        :return: None
        '''

        noise = np.random.normal(0, freq_var, size=(self.fids.shape[0], self.fids.shape[1], self.transient_count))
        self.fids = self.fids * np.exp(1j * self.t.reshape(1, -1, 1) * noise * 2 * math.pi)


    def add_freq_shift_linear(self, freq_offset_var=0, freq_slope_var=5):
        '''
        Adds a random number of linear frequency shifts of different transient lengths throughout the scan.

        :param freq_offset_var: standard deviation of the frequency offset term of the distribution of the signal.
        :param freq_slope_var: standard deviation of the frequency slope term of the distribution of the signal.
        :return: None
        '''

        # select random number of sets of linear frequency shifts
        num_sets = int(np.random.uniform(0, 3))

        for sets in range(0, num_sets):
            # select where and for how long the linear frequency drift will last
            numTrans = int(np.random.uniform(5, (self.transient_count / 2)))
            startTrans = int(np.random.uniform(0, (self.transient_count / 2)))
            endTrans = startTrans + numTrans

            offset_noise = np.random.normal(0, freq_offset_var, size=(self.fids.shape[0], 1, 1))
            slope_noise = np.random.normal(0, freq_slope_var, size=(self.fids.shape[0], 1, numTrans))
            noise = offset_noise + np.repeat(np.arange(0, 1, 1 / numTrans).reshape(1, 1, -1), self.fids.shape[0], axis=0) * slope_noise

            for trans in range(0, numTrans):
                self.fids[:, :, startTrans:endTrans] = self.fids[:, :, startTrans:endTrans] * np.exp(1j * self.t.reshape(1, -1, 1) * noise[:, :, :] * 2 * math.pi)


    def add_phase_shift_random(self, phase_var=20):
        '''
        Adds random phase shifts (with normal distribution) to the FID signal.

        :param phase_var: standard deviation of the phase shifts distribution of the signal.
        :return: None
        '''

        noise = np.random.normal(0, phase_var, size=(self.fids.shape[0], self.fids.shape[1], self.transient_count))
        self.fids = self.fids * np.ones(self.fids.shape) * np.exp(1j * noise * math.pi / 180)


    ########################################################################################################################
    # Artifact functions
    ########################################################################################################################
    def add_ghost_artifact(self, nmb_ghost, start, stop, amp, mult, cs):
        '''
        Adds a ghost (frequency domain) or spurious echo (time domain) artifact to a subset of transients.
        Adapted from Supplementary Material from (Berrington et al. 2021)
        Param defaults are set within the function to maximize variability.

        Defaults are random uniform distributions.
        :param nmb_ghost: number of ghost artifacts to be added.
        :param start: time value (within the FID) to start the echo.
        :param stop: time value (within the FID) to end the echo.
        :param amp: amplitude of the echo.
        :param mult: pi multiplier to assign a (base) phase to the echo.
        :param cs: chemical shift value (in ppm) for the artifact in the frequency domain.
        :return: None
        '''

        # select random FIDs/Transients to insert ghost artifact
        if nmb_ghost == None:
            nmb_ghost = int(np.random.uniform(int(self.transient_count / 5), int(self.transient_count / 2)))
        self.ghost_locations = np.sort(np.random.choice(range(0, self.transient_count), size=nmb_ghost, replace=False))

        for i in self.ghost_locations:
            # select random location in FID (time domain) to place the artifact
            time = self.t.reshape(1, -1, 1)
            if start == None:
                start = int(np.random.uniform(700, 1200))  # start at 0.35-0.5 seconds
            if stop == None:
                stop = int(start + np.random.uniform(100, 500))  # end at 0.6-0.75 seconds

            # variables related to the artifact: amplitude, phase, T_all, larmor frequency, and chemical shift artifact (+/- ppm)
            if amp == None:
                amp = np.random.uniform(100, 400, size=1)
            if mult == None:
                mult = np.random.uniform(0.1, 1.9, size=1)
            phase = mult * math.pi
            T_all = np.max(self.t)

            # cs spectrum/user values [0(3), 1(2), 2(1), 3(0), 4(-1), 5(-2), 6(-3)] map to ghost equation/actual values [3, 2, 1, 0, -1, -2, -3]
            lf = 2 * np.pi * 127
            if cs == None:
                cs = np.random.uniform(-3.0, 3.0, size=1)

            # insert ghost artifact into FID
            on_or_off = int(np.random.uniform(0, 1))
            self.fids[on_or_off, start:stop, i] = self.fids[on_or_off, start:stop, i] * (amp * np.exp(-abs(time[0, start:stop, 0]) / T_all) * np.exp(1j * (lf * (1 - cs) * time[0, start:stop, 0] + phase)))


    def add_EddyCurrent_artifact(self, nmb_ec, amp, tc):
        '''
        Adds an Eddy Current artifact to a subset of transients.
        Adapted from Material from FID-A (Simpson et al, 2017)
        Param defaults are set within the function to maximize variability.

        Defaults are random uniform distributions.
        :param nmb_ec: number of EC artifacts to be added.
        :param amp: amplitude of the EC.
        :param tc: time constant of the decaying EC.
        :return: None
        '''

        # select random FIDs/Transients to insert EC artifact
        if nmb_ec == None:
            nmb_ec = int(np.random.uniform(int(self.transient_count / 5), int(self.transient_count / 2)))
        self.EC_locations = np.sort(np.random.choice(range(self.transient_count), size=nmb_ec, replace=False))

        # variables related to the artifact: amplitude and time constant of decaying artifact (in seconds)
        if amp == None:
            amp = np.random.uniform(1.000, 1.500, size=1)
        if tc == None:
            tc = np.round(np.random.uniform(0.080, 0.120, size=1), 3)

        # insert EC artifact into FID
        for i in self.EC_locations:
            on_or_off = int(np.random.uniform(0, 1))
            self.fids[on_or_off, :, i] = self.fids[on_or_off, :, i] * (np.exp(-1j * self.t * (amp * np.exp(self.t / tc)) * 2 * math.pi))


    def motion_lineBroad_artifact(self, nmb_motion, amp, lvs):
        '''
        Adds a line broadening (LB) artifact (for large motion) to a subset of transients.

        Defaults are random uniform distributions.
        :param nmb_motion: number of LB artifacts to be added.
        :param amp: amplitude of the LB artifact.
        :param lvs: lineshape variance of the LB artifact.
        :return: None
        '''

        # select random FIDs/Transients to insert large motion artifact
        if nmb_motion == None:
            nmb_motion = int(np.random.uniform(int(self.transient_count / 5), int(self.transient_count / 2)))
        self.motion_locations = np.sort(np.random.choice(range(self.transient_count), size=nmb_motion, replace=False))

        # variables related to the artifact: amplitude and lineshape variance
        if amp == None:
            amp = np.round(np.random.uniform(1.500, 2.250, size=1), 3)
        if lvs == None:
            lvs = np.round(np.random.uniform(20, 40, size=1), 2)
        self.motion_locations = self.motion_locations

        # insert LB artifact into FID
        for i in self.motion_locations:
            self.fids[0, :, i] = self.fids[0, :, i] * (amp * np.exp(-self.t * lvs))
            self.fids[1, :, i] = self.fids[1, :, i] * (amp * np.exp(-self.t * lvs))


    def motion_baseline_artifact(self):
        '''
        Adds a wandering baseline artifact to the same transients as identified by self.motion_locations in the motion_lineBroad_artifact function.
        The following attributes need to be defined: self.motion_locations

        :return: None
        '''

        # get specs
        on_specs, off_specs = self.get_specs()
        num_bases = []

        # get the number of sine baselines per motion corrupted transient
        for i in range(self.motion_locations.shape[0]):
            p = random.randint(1, 4)
            num_bases.append(p)

        # generate the motion corrupted baseline and insert into transient
        dc_shift = np.argmax(self.ppm <= -1.0)
        for i in range(0, self.motion_locations.shape[0]):
            amp, cyc = np.random.uniform(0.05, 0.10, size=num_bases[i]), np.random.uniform(0.1, 3, size=num_bases[i])

            for k in range(0, num_bases[i]):
                frame_start_ppm = np.argmax(self.ppm <= np.random.uniform(self.ppm[0], self.ppm[1024]))
                frame_end_ppm = np.argmax(self.ppm <= np.random.uniform(self.ppm[1024], self.ppm[2047]))

                # creation of baseline contamination for ON
                base = np.zeros(len(self.ppm))
                sine = abs(amp[k] * np.sin(cyc[k] * self.ppm[frame_start_ppm:frame_end_ppm]))
                base[frame_start_ppm:frame_end_ppm] = signal.savgol_filter(sine, 201, 5, mode='nearest')
                on_specs[0, :, self.motion_locations[i]] = base + on_specs[0, :, self.motion_locations[i]] - on_specs[0, dc_shift, self.motion_locations[i]]

                # creation of baseline contamination for OFF
                base = np.zeros(len(self.ppm))
                varAmp, varCyc = np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01)
                sine = abs(varAmp * amp[k] * np.sin(varCyc * cyc[k] * self.ppm[frame_start_ppm:frame_end_ppm]))
                base[frame_start_ppm:frame_end_ppm] = signal.savgol_filter(sine, 201, 5, mode='nearest')
                off_specs[0, :, self.motion_locations[i]] = base + off_specs[0, :, self.motion_locations[i]] - off_specs[0, dc_shift, self.motion_locations[i]]

        # return to fids
        self.fids[self.edit_on == 1] = to_fids(on_specs)
        self.fids[self.edit_on == 0] = to_fids(off_specs)


    def lipid_peak_contamination(self, nmb_lp):
        '''
        Adds a lipid peak contamination between 1.5 and 1.6 ppm to a random subset of transients.

        :param nmb_lp: number of lipid peak artifacts to be added.
        :return: None
        '''

        # get specs and location of lipid contaminations
        on_specs, off_specs = self.get_specs()
        if nmb_lp == None:
            nmb_lp = int(np.random.uniform(int(self.transient_count / 5), int(self.transient_count / 2)))
        self.LP_locations = np.sort(np.random.choice(range(self.transient_count), size=nmb_lp, replace=False))

        for i in range(0, self.LP_locations.shape[0]):
            glob_loc = np.random.uniform(1.50, 1.60)
            # on spectra - contamination created as two overlapping Gaussians (randomize parameters as much as possible)
            amp1, loc1, sig1 = np.random.uniform(0.35, 0.24), np.random.uniform(glob_loc + 0.05, glob_loc + 0.07), np.random.uniform(0.02, 0.10)
            amp2, loc2, sig2 = np.random.uniform(0.05, 0.10), np.random.uniform(glob_loc - 0.07, glob_loc - 0.05), np.random.uniform(0.02, 0.10)
            gauss_p1 = amp1 * (np.exp(-np.power(self.ppm - loc1, 2.) / (2 * np.power(sig1, 2.))))
            gauss_p2 = amp2 * (np.exp(-np.power(self.ppm - loc2, 2.) / (2 * np.power(sig2, 2.))))
            gauss_final_on = gauss_p1 + gauss_p2
            on_specs[0, :, self.LP_locations[i]] = on_specs[0, :, self.LP_locations[i]] + gauss_final_on

            # off spectra - contamination created as two overlapping Gaussians (randomize parameters as much as possible)
            amp3, loc3, sig3 = np.random.uniform(0.05, 0.10), np.random.uniform(glob_loc + 0.05, glob_loc + 0.07), np.random.uniform(0.02, 0.10)
            amp4, loc4, sig4 = np.random.uniform(0.01, 0.05), np.random.uniform(glob_loc - 0.07, glob_loc - 0.05), np.random.uniform(0.02, 0.10)
            gauss_p3 = amp3 * (np.exp(-np.power(self.ppm - loc3, 2.) / (2 * np.power(sig3, 2.))))
            gauss_p4 = amp4 * (np.exp(-np.power(self.ppm - loc4, 2.) / (2 * np.power(sig4, 2.))))
            gauss_final_off = gauss_p3 + gauss_p4
            off_specs[0, :, self.LP_locations[i]] = off_specs[0, :, self.LP_locations[i]] + gauss_final_off

        # return to fids
        self.fids[self.edit_on == 1] = to_fids(on_specs)
        self.fids[self.edit_on == 0] = to_fids(off_specs)


    def lipid_baseline_contamination(self):
        '''
        Adds a baseline contamination caused by the lipid contamination between 1.5 and 1.6 ppm to the same
        transients as identified by self.LP_locations in the lipid_peak_contamination function.
        The following attributes need to be defined: self.LP_locations.

        :return: None
        '''

        # get specs and randomize baseline contamination
        on_specs, off_specs = self.get_specs()
        ampMin, ampMaj = np.random.uniform(0.05, 0.10, size=self.transient_count), np.random.uniform(0.10, 0.20, size=self.transient_count)
        cycMin, cycMaj = np.random.uniform(0.90, 0.95, size=self.transient_count), np.random.uniform(1.20, 1.25, size=self.transient_count)

        for i in range(0, len(self.LP_locations)):
            # creation of baseline contamination
            baseMaj, baseMin = np.zeros(len(self.ppm)), np.zeros(len(self.ppm))
            frame_000ppm, frame_198ppm, frame_200ppm, frame_303ppm = np.argmax(self.ppm <= 0.0), np.argmax(self.ppm <= 1.98), np.argmax(self.ppm <= 2.00), np.argmax(self.ppm <= 3.03)
            sineMaj, sineMin = abs(ampMaj[i] * np.sin(cycMaj[i] * self.ppm[frame_198ppm:frame_000ppm])), abs(ampMin[i] * np.cos((cycMin[i] * self.ppm[frame_303ppm:frame_200ppm])))
            baseMin[frame_303ppm:frame_200ppm], baseMaj[frame_198ppm:frame_000ppm] = sineMin, sineMaj
            baseMin, baseMaj = baseMin, baseMaj
            on_specs[0, :, self.LP_locations[i]] = baseMin + baseMaj + on_specs[0, :, self.LP_locations[i]]

        # return to fids
        self.fids[self.edit_on == 1] = to_fids(on_specs)
        self.fids[self.edit_on == 0] = to_fids(off_specs)


    ########################################################################################################################
    # Metric functions
    ########################################################################################################################
    def get_SNR(self, specs, single='no'):
        '''
        Get GABA signal-to-noise ratio (SNR).
        Signal-to-Noise Ratio (Mikkelsen et al. (2017) - Big GABA: Edited MRS at 24 research sites)

        :param specs: frequency domain difference transients in form.
        :param single: specifies whether a single (mean) transient or set (scan) of transients is passed.
        :return: returns the GABA SNR.
        '''

        # select window and specs set (diff vs. OFF) based on metabolite selected
        metab_indClose, metab_indFar = np.amax(np.where(self.ppm >= 2.8)), np.amin(np.where(self.ppm <= 3.2))
        if single == 'yes':
            mean_specs = np.real(specs.flatten())
        else:
            mean_specs = np.real(specs.mean(axis=2).flatten())
        noise_indClose, noise_indFar = np.amax(np.where(self.ppm >= 10.0)), np.amin(np.where(self.ppm <= 11.0))
        noise_ppm = self.ppm[noise_indFar:noise_indClose].shape[0]

        # SNR of Individual Transients
        all_MAXy = np.amax(mean_specs[metab_indFar:metab_indClose], axis=0)
        allSNR_dt = np.polyfit(self.ppm[noise_indFar:noise_indClose], mean_specs[noise_indFar:noise_indClose], 2)
        allSNR_stdev = abs(np.sqrt(np.sum(np.square(((mean_specs[noise_indFar:noise_indClose]) - np.polyval(allSNR_dt[:], self.ppm[noise_indFar:noise_indClose])))) / (noise_ppm - 1)))
        allSNR = (all_MAXy) / (2 * allSNR_stdev)

        return allSNR


    def get_LW(self, specs, single='no'):
        '''
        Measures the GABA linewidth.

        :param specs: frequency domain difference transients in form.
        :param single: specifies whether a single (mean) transient or set (scan) of transients is passed.
        :return: returns the GABA linewidth in hertz.
        '''

        # select window and specs set (diff vs. OFF) based on metabolite selected
        metab_indClose, metab_indFar = np.amax(np.where(self.ppm >= 2.9)), np.amin(
        np.where(self.ppm <= 3.1))  # was 2.8 to 3.2
        if single == 'yes':
            mean_specs = np.real(specs.flatten())
        else:
            mean_specs = np.real(specs.mean(axis=2).flatten())

        # calculate fwhm
        global LHM_x, RHM_x
        MAXy = np.amax(mean_specs[metab_indFar:metab_indClose], axis=0)
        Metab_x = mean_specs[metab_indFar:metab_indClose]
        MetabMAX_x = np.array(np.argmax(Metab_x, axis=0))
        LPB_x, RPB_x = Metab_x[:MetabMAX_x], Metab_x[MetabMAX_x:]
        try:
            LHM_x, RHM_x = np.amin(np.where(LPB_x > (MAXy / 2))) + metab_indClose, np.amax(
                np.where(RPB_x > (MAXy / 2))) + metab_indClose + MetabMAX_x
        except ValueError:  # raised if `y` is empty.
            pass
        ToPPM_L, ToPPM_R = self.ppm[LHM_x], self.ppm[RHM_x]

        FWHM = (np.sqrt(2 * np.log(2)) * np.abs(ToPPM_L - ToPPM_R)) * 127.7  # to obtain linewidth in hertz, assuming the precessional F is 127.7 Hz

        return FWHM


    def get_ShapeScore(self, specs, gt, single='No'):
        '''
        Calculates the shape score of GABA and GLX.
        Shape score as defined in (Berto et al. (2023) - Advancing GABA-Edited MRS Through a Reconstruction Challenge)

        :param specs: frequency domain difference transients in form.
        :param gt: passes the ground truth difference spectrum (frequency domain).
        :param single: specifies whether a single (mean) transient or set (scan) of transients is passed.
        :return: returns the shape score in percentage (decimal) form.
        '''

        # Select windows for metabolites
        gaba_indFar, gaba_indClose = np.amax(np.where(self.ppm >= 2.8)), np.amin(np.where(self.ppm <= 3.2))
        GLX_indFar, GLX_indClose = np.amax(np.where(self.ppm >= 3.6)), np.amin(np.where(self.ppm <= 3.9))

        # Calculate correlation coefficient between simulated transients and ground truths
        if single == 'yes':
            specs = np.real(specs.flatten())
        else:
            specs = np.real(specs.mean(axis=2).flatten())

        gaba_specs, glx_specs = specs[gaba_indClose:gaba_indFar], specs[GLX_indClose:GLX_indFar]
        gaba_specs, glx_specs = (gaba_specs - gaba_specs.min()) / (gaba_specs.max() - gaba_specs.min()), (glx_specs - glx_specs.min()) / (glx_specs.max() - glx_specs.min())
        gaba_gt, glx_gt = np.real(gt[gaba_indClose:gaba_indFar]), np.real(gt[GLX_indClose:GLX_indFar])
        gaba_gt_1, glx_gt_1 = (gaba_gt - gaba_gt.min()) / (gaba_gt.max() - gaba_gt.min()), (glx_gt - glx_gt.min()) / (glx_gt.max() - glx_gt.min())
        SS = np.corrcoef(gaba_specs, gaba_gt_1)[0, 1] * 0.6 + np.corrcoef(glx_specs, glx_gt_1)[0, 1] * 0.4

        return SS


    def get_OutlierScore(self, specs, single='No'):
        '''
        Calculates the percentage of points within a window which are outliers compared to the mean (based on 3 standard deviations rule)

        :param specs: frequency domain difference transients in form.
        :param single: specifies whether a single (mean) transient or set (scan) of transients is passed.
        :return: returns the outlier score in percentage (decimal) form.
        '''

        mean_curve = self.nonArtifTrans.mean(axis=2).flatten()
        # Window Selection
        indClose = np.amin(np.where(self.ppm <= 1.09))
        indFar = np.amin(np.where(self.ppm <= 5.00))
        nonArtifTrans = self.nonArtifTrans[:, indFar:indClose, :]

        if single == 'yes':
            specs = specs[:, indFar:indClose]
            outDetect = 0

            for specPoints in range(0, specs.shape[1]):
                specVal = np.real(specs[0, specPoints])
                meanValUp = np.real(mean_curve[specPoints]) + np.real((3 * np.std(abs(nonArtifTrans[0, specPoints, :]))))
                meanValDown = np.real(mean_curve[specPoints]) - np.real((3 * np.std(abs(nonArtifTrans[0, specPoints, :]))))

                if not ((specVal < meanValUp) and (specVal > meanValDown)):
                    outDetect = outDetect + 1

            outScore = (outDetect / specs.shape[1])


        else:
            specs = specs[:, indFar:indClose, :]
            outScore = [None] * specs.shape[2]

            for trans in range(0, specs.shape[2]):
                outDetect = 0

                for specPoints in range(0, specs.shape[1]):
                    specVal = np.real(specs[0, specPoints, trans])
                    meanValUp = np.real(mean_curve[specPoints]) + np.real((3 * np.std(abs(nonArtifTrans[0, specPoints, :]))))
                    meanValDown = np.real(mean_curve[specPoints]) - np.real((3 * np.std(abs(nonArtifTrans[0, specPoints, :]))))

                    if not ((specVal < meanValUp) and (specVal > meanValDown)):
                        outDetect = outDetect + 1

                outScore[trans] = (outDetect / specs.shape[1])

        return outScore


    ########################################################################################################################
    # Domain functions
    ########################################################################################################################
    def reset_fids(self):
        '''
        Resets FIDs to their original (import/ground truth) form.

        :return: None
        '''

        self.fids = self.base_fids_on.repeat(2, axis=0) * self.edit_on.reshape(-1, 1) + self.base_fids_off.repeat(2, axis=0) * (1 - self.edit_on.reshape(-1, 1))
        self.fids = self.fids.reshape(self.fids.shape[0], self.fids.shape[1], 1).repeat(self.transient_count, axis=2)


    def get_fids(self):
        '''
        Provides the ON and OFF FIDs (time domain) in their current state.

        :return: None
        '''

        return self.fids[self.edit_on == 1], self.fids[self.edit_on == 0]


    def get_specs(self):
        '''
        Provides the ON and OFF SPECs (frequency domain) in their current state.

        :return: None
        '''

        on_fids = self.fids[self.edit_on == 1]
        off_fids = self.fids[self.edit_on == 0]

        on_spec = to_specs(on_fids)
        off_spec = to_specs(off_fids)

        return on_spec, off_spec


    def get_differenceSpecs(self):
        '''
        Provides the difference SPECs (ON-OFF) (frequency domain) in their current state.

        :return: None
        '''

        return to_specs(self.fids[self.edit_on == 1]) - to_specs(self.fids[self.edit_on == 0])


    def get_differenceFids(self):
        '''
        Provides the difference FIDs (ON-OFF) (time domain) in their current state.

        :return: None
        '''

        return self.fids[self.edit_on == 1] - self.fids[self.edit_on == 0]


    def insert_corrupt(self, corruptTransients):
        '''
        Inserts corrupt transients at random locations whithin the typical scan.
        The following attributes need to be defined: self.artifact_locations, *attributes for corrupt transient maker object.

        :param corruptTransients: corrupt transient object.
        :return: ON and OFF Specs (Frequency domain)
        '''

        # get both typical and corrupted specs
        on_specs, off_specs = self.get_specs()
        on_corr_specs, off_corr_specs = corruptTransients.get_specs()

        # Select random locations and insert (with replacement) the corrupted transients in the typical scan
        self.artifact_locations = np.sort(np.random.choice(range(self.transient_count), size=corruptTransients.transient_count, replace=False))
        for i in range(0, corruptTransients.transient_count):
            on_specs[0, :, self.artifact_locations[i]] = on_corr_specs[0, :, i]
            off_specs[0, :, self.artifact_locations[i]] = off_corr_specs[0, :, i]

        # Get new locations of artifacts (within the typical scan)
        if any(corruptTransients.ghost_locations):
            for i in range(corruptTransients.ghost_locations.shape[0]):
                corruptTransients.ghost_locations[i] = self.artifact_locations[corruptTransients.ghost_locations[i]]
        if any(corruptTransients.EC_locations):
            for j in range(corruptTransients.EC_locations.shape[0]):
                corruptTransients.EC_locations[j] = self.artifact_locations[corruptTransients.EC_locations[j]]
        if any(corruptTransients.motion_locations):
            for k in range(corruptTransients.motion_locations.shape[0]):
                corruptTransients.motion_locations[k] = self.artifact_locations[corruptTransients.motion_locations[k]]
        if any(corruptTransients.LP_locations):
            for i in range(corruptTransients.LP_locations.shape[0]):
                corruptTransients.LP_locations[i] = self.artifact_locations[corruptTransients.LP_locations[i]]

        self.ghost_locations, self.EC_locations, self.motion_locations, self.LP_locations = corruptTransients.ghost_locations, corruptTransients.EC_locations, corruptTransients.motion_locations, corruptTransients.LP_locations

        # return to fids
        self.fids[self.edit_on == 1] = to_fids(on_specs)
        self.fids[self.edit_on == 0] = to_fids(off_specs)

        return on_specs, off_specs


def to_fids(in_specs):
    '''
    Convert SPECs (frequency domain) to FIDs (time domain).

    :param in_specs: SPECs
    :return: FIDs
    '''

    return np.fft.fft(np.fft.fftshift(in_specs, axes=1), axis=1)


def to_specs(in_fids):
    '''
    Convert FIDs (time domain) to SPECs (frequency domain).

    :param in_fids: FIDs
    :return: SPECs
    '''
    return np.fft.fftshift(np.fft.ifft(in_fids, axis=1), axes=1)
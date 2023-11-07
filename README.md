# GABA-Edited-Artifact-Simulation-Toolbox
GABA-Edited artifact simulation toolbox for simulating spurious echoes (ghosting artifact), eddy currents, lipid contamination and different degrees of motion contamination.

# Basic Usage
Example usage can be found in the ArtifactToolboxExample notebook.

# transient_maker Class
Class functions are divided by usage:

Domain Functions\
reset_fids(), get_fids(), get_specs(), get_differenceSpecs(), get_differenceFids(), to_fids(), to_specs(), and insert_corrupt()

Typical Noise Functions\
add_time_domain_noise(): add Gaussian white (amplitude) noise with user standard deviation.\
add_phase_shift_random(): add random normal phase shifts with user standard deviation to mimic small motion.\
add_frequency_shift_random(): add random normal frequency shifts with user standard deviation to mimic small motion.\
add_freq_shift_linear(): add  

Artifact Functions\
add_ghost_artifact(): add spurious echo artifact with user input for number, start and end time of the echo, amplitude, phase and chemical shift of the artifact.\
add_EddyCurrent_artifact(): add eddy current artifact with user input for number, amplitude and time constant of the artifact.\
motion_lineBroad_artifact(): add line broadening artifact associated with large motion with user input number, amplitude and lineshape variance of the artifact.\
motion_baseline_artifact(): add whole baseline contamination associated with large motion.\
lipid_peak_contamination(): add lipid peak contamination with user input for number of artifacts.\
lipid_baseline_contamination(): add constrained baseline contamination associated with lipid contamination.

Metric Functions\
get_SNR(): calculates the SNR of the GABA peak from the difference spectrum. Can be applied to a single difference transient or list of difference transients.\
get_LW(): calculates the GABA linewidth (FWHM) from the difference spectrum. Can be applied to a single difference transient or list of difference transients.\
get_ShapeScore(): calculates the correlation between the GABA and GLX peak from the ground truth and degraded difference transient as per Berto et al (2023).\
get_OutlierScore(): calculates, for a single difference transient, the percentage of points within a frequency window that are 3 standard deviations from the mean.

Output Summary Functions\
print_results()

# Display File
display_artifact(): graphs the first difference transient containing the requested artifact in the frequency and time domains along with their ground truths.\
display_transToDate(): graphs the difference transients in their current state in the frequency domain with the corresponding ground truth.\
saveData(): saves the frequency and time domain complex data to an .npy file.

# GABA-Edited-Artifact-Simulation-Toolbox
GABA-Edited artifact simulation toolbox for simulating spurious echoes (ghosting artifact), eddy currents, lipid contamination and different degrees of motion contamination ('subtle', 'progressive', and 'disruptive').\
The GettingStarted notebook, fids, ppm, and time files can be used to sample the toolbox.\
The gaba_edited_artifact_simulation_toolbox python file contains all functions presented below.

# Basic Usage
Example usage can be found in main.py.\
For adding artifacts:\
1.) Load ground truths (GTs).\
2.) Simulate a scan with arbitrary number of ON and OFF pairs.\
3.) Add selected artifacts.\
4.) Visualize artifacts and save data.

# Artifact Functions
add_ghost_artifact(): add spurious echo artifact with user input for number, start and end time of the echo, amplitude, phase and chemical shift of the artifact.\
add_eddy_current_artifact(): add eddy current artifact with user input for number, amplitude and time constant of the artifact.\
add_subtle_motion_artifact(): add small random frequency and phase shifts constrained within a range.\
add_progressive_motion_artifact(): add linear frequency shift over multiple consecutive scans to simulate participant falling asleep with head drift.\
add_disruptive_motion_artifact(): add line broadening and baseline contamination associated with large scale motion such as coughing.\
add_lipid_artifact(): add lipid peak contamination and constrained baseline associated with lipid contamination.

# Helper Functions
to_fids(): converts specs to time domain fids.\
to_specs(): converts fids to frequency domain specs.\
interleave(): interleaves data to be in order of collection.\
undo_interleave(): undoes interleaving to have distinct ON and OFF transient arrays.\
scale(): scales the data to work with default values.\
undo_scale(): undoes the scaling done to the data with scaling factor.\

# Artifact Simulation Toolbox by Hanna Bugler, Rodrigo Berto, Roberto Souza and Ashley Harris (2023)
# Graphs to display artifacts, transients to-date and to save spec and fid data

import matplotlib.pyplot as plt
import numpy as np

def display_artifact(diff_gt_fid, diff_gt_spec, transientObject, artifact_name=None, location=[]):
# code to display individual transient artifact (by default, displays first artifact in location array)
    # get difference FID and Spec
    on_corr_specs, off_corr_specs = np.copy(transientObject.get_specs())
    on_corr_specs, off_corr_specs = on_corr_specs[0, :, location[0]], off_corr_specs[0, :, location[0]]
    spec = np.real(on_corr_specs - off_corr_specs).squeeze()

    on_corr_fids, off_corr_fids = np.copy(transientObject.get_fids())
    on_corr_fids, off_corr_fids = on_corr_fids[0, :, location[0]], off_corr_fids[0, :, location[0]]
    fid = np.real(on_corr_fids - off_corr_fids).squeeze()

    # top: after time domain
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(transientObject.t, np.real(diff_gt_fid), 'k', linewidth=1, label='Ground Truth FID', alpha=0.75)
    ax1.plot(transientObject.t, fid, 'r', linewidth=1, label='Corrupt FID', alpha=0.75)
    ax1.set_title(f"Time domain FID AFTER {artifact_name} Applied")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(np.min(fid)-5, np.max(fid)+5)
    ax1.get_yaxis().set_visible(False)
    ax1.legend(bbox_to_anchor=(1, 1))

    # bottom: after frequency domain
    ax2.plot(transientObject.ppm, np.real(diff_gt_spec), 'k', linewidth=1, label='Ground Truth Spectrum', alpha=0.75)
    ax2.plot(transientObject.ppm, spec, 'b', linewidth=1, label='Corrupt Spectrum', alpha=0.75)
    ax2.set_title(f"Frequency Domain Spectrum AFTER {artifact_name} Applied")
    ax2.set_xlim(0.0, 5.0)
    ax2.invert_xaxis()
    # ax2.set_ylim(np.min(spec)+0.5, np.max(spec)+0.5)
    ax2.get_yaxis().set_visible(False)
    ax2.legend(bbox_to_anchor=(1, 1))
    plt.show()


def display_transToDate(transientObject, gt=None):
# code to display transients to date
    # get difference spec
    on_specs, off_specs = transientObject.get_specs()
    diff_specs = np.real(on_specs - off_specs)

    # plot individual transients
    fig, ax = plt.subplots()
    for i in range(0, diff_specs.shape[2]):
        ax.plot(transientObject.ppm, diff_specs[0, :, i], linewidth=1, alpha=0.5)

    # PLOT OVER THE INDIVIDUAL TRANSIENTS THE MEAN SPECTRUM AND GROUND TRUTH SPECTRUM
    ax.plot(transientObject.ppm, np.real(gt), 'k', linewidth=2, label='Ground Truth Spectrum')
    ax.plot(transientObject.ppm, diff_specs.mean(axis=2).flatten(), 'r', linewidth=2, label='Mean Spectrum')
    ax.set_title("Transients-to-Date")
    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(np.min(diff_specs)-0.1, np.max(diff_specs)+0.1)
    ax.set_ylim(np.min(diff_specs)-0.05, np.max(diff_specs)+0.2)
    ax.invert_xaxis()
    ax.get_yaxis().set_visible(False)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()


def saveData(all_specs, all_fids):
    # save specs and quality labels to .npy file
    np.save("SPECS.npy", all_specs)
    np.save("FIDS.npy", all_fids)

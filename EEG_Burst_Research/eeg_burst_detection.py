#!/usr/bin/env python3


import os
import numpy as np
import scipy.io
import wfdb
from scipy.signal import butter, filtfilt
import resampy
import matplotlib.pyplot as plt


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_pt_from_fname(filepath):
    """
    Extract patient ID from filename.
    
    Args:
    - filepath (str): Path to the file.
    
    Returns:
    - patient_id (str): Extracted patient ID.
    """
    filename = os.path.basename(filepath)
    
    # Assuming the patient ID is the second part of the filename, split by '_'
    parts = filename.split('_')
    
    if len(parts) > 1:
        patient_id = parts[1]  # Extracts '0284' from 'ICARE_0284_05'
    else:
        raise ValueError("Filename format is incorrect.")
    
    return patient_id


def extract_header(filename):
    """Extract header information from EEG file."""
    record = wfdb.rdheader("0284_005_008_EEG")
    fs = record.fs
    channel = record.sig_name
    return fs, channel


def save_eeg_data(eeg_data, save_path):
    """
    Saves EEG data to a file.
    
    Args:
    - eeg_data (np.array): Processed EEG data.
    - save_path (str): Path to save the data.
    """
    np.save(save_path, eeg_data)


# =============================================================================
# EEG PREPROCESSING FUNCTIONS
# =============================================================================

def reorder_eeg_channels(eeg_data, channel_names):
    """Reorder EEG channels to standard montage."""
    desired_channel_order = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                             'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    mapping_dict = {}
    reordered_channels_data = []

    patient_eeg = eeg_data  # Shape: [n_channels, n_samples]
    patient_channels = channel_names  # List or array of channel names

    n_samples = patient_eeg.shape[1]  # Number of samples per channel

    for channel in desired_channel_order:
        if channel in patient_channels:
            index = patient_channels.index(channel)
            mapping_dict[channel] = index
            channel_data = patient_eeg[index]  # Extract the channel data
            reordered_channels_data.append(channel_data)
        else:
            # If the desired channel is not present, fill with zeros
            mapping_dict[channel] = None
            channel_data = np.zeros(n_samples)
            reordered_channels_data.append(channel_data)
            print(str(channel) + " is missing, so it was padded with zeros")

    # Convert the list to a NumPy array
    reordered_eeg_data = np.array(reordered_channels_data)

    return reordered_eeg_data, mapping_dict


def standardize_struct(input_data, fs, channel):
    """
    Standardize EEG data structure by unifying channel nomenclature, reordering channels, and downsampling.
    
    Args:
    - input_data: Input EEG data
    - fs: Original sampling frequency
    - channel: Channel names
    
    Returns:
    - output: Standardized EEG data structure.
    - channel_mapping: Channel mapping dictionary
    """
    # Constants
    FREQUENCY = fs  # Assumes same frequency for all channels
    N_CHANNELS = 19  # Final number of channels
    TARGET_FREQUENCY = 200  # Target frequency for downsampling

    # Unify nomenclature and reorder channels
    reordered_eeg, channel_mapping = reorder_eeg_channels(input_data, channel)

    # Downsample to target frequency
    output = resampy.resample(reordered_eeg, TARGET_FREQUENCY, FREQUENCY)

    return output, channel_mapping


def load_mat_eeg(fs, channel_order, filepath):
    """
    Load EEG data from .mat files and process it.
    
    Args:
    - fs (float): Sampling rate
    - channel_order (list): List of channel names
    - filepath (str): Path to the .mat file.
    
    Returns:
    - eeg_data (np.array): Processed EEG data.
    - channel_mapping (dict): Channel mapping dictionary
    """
    print(f"Processing .mat file from {filepath}")
    eeg_struct = scipy.io.loadmat(filepath)
    eeg_data = eeg_struct.get('val', None)  
    
    if eeg_data is None:
        raise ValueError("EEG data not found in the .mat file.")
    
    # Standardize the EEG structure
    eeg_data, channel_mapping = standardize_struct(eeg_data, fs, channel_order)
    
    print("Channel mapping:", channel_mapping)
    
    return eeg_data, channel_mapping


def check_loaded_eeglab(eeg_data):
    """
    Check if loaded EEG data is valid.
    
    Args:
    - eeg_data (np.array): Loaded EEG data.
    
    Returns:
    - is_good (bool): True if EEG data is valid, False otherwise.
    """
    if eeg_data is not None and len(eeg_data.shape) == 2:
        nchans, ns = eeg_data.shape
        if nchans > 0 and ns > 0:
            print("Loaded EEG data seems good.")
            return True
    print("Invalid EEG data.")
    return False


def preprocess(eeg_data, srate):
    """
    Preprocess the EEG data by applying filters.
    
    Args:
    - eeg_data (np.array): EEG data as a NumPy array (channels x samples).
    - srate (int): Sampling rate of the EEG data.
    
    Returns:
    - eeg_preprocessed (np.array): Preprocessed EEG data.
    """
    hp_filter_order = 4 
    low_freq_threshold = 0.5  
    lp_filter_order = 4  
    high_freq_threshold = 50  
    notch_filter_order = 4  
    notch_low_threshold = 59  
    notch_high_threshold = 61  

    bh, ah = butter(hp_filter_order, low_freq_threshold / (srate / 2), 'high')  # High-pass filter
    bl, al = butter(lp_filter_order, high_freq_threshold / (srate / 2), 'low')  # Low-pass filter
    bn, an = butter(notch_filter_order, 
                    [notch_low_threshold / (srate / 2), notch_high_threshold / (srate / 2)], 
                    'bandstop')  # Notch filter
    data = eeg_data.T

    # Applies the filters using filtfilt (zero-phase filtering)
    data = filtfilt(bh, ah, data, axis=0)  # High-pass filter
    data = filtfilt(bn, an, data, axis=0)  # Notch filter
    data = filtfilt(bl, al, data, axis=0)  # Low-pass filter

    # Transposes back to channels x samples
    eeg_preprocessed = data.T

    return eeg_preprocessed


# =============================================================================
# BURST SUPPRESSION DETECTION FUNCTIONS
# =============================================================================

def convert_indices_to_index_ranges(indices, min_gap=2):
    """
    Group consecutive indices into [start, end] ranges.

    Parameters
    ----------
    indices : 1D array-like of sorted integer indices
    min_gap : int
        The minimum jump between successive indices to consider them "disconnected."
        If min_gap=2, then any pair of consecutive indices that differ by more than 1
        starts a new range.

    Returns
    -------
    ranges : np.ndarray
        2D array of shape (N, 2), where each row is [start_idx, end_idx].
    """
    if len(indices) == 0:
        return np.zeros((0, 2), dtype=int)

    ranges = []
    start_idx = indices[0]
    prev_idx = indices[0]

    for i in indices[1:]:
        # If there's a break in consecutive indices
        if (i - prev_idx) >= min_gap:
            # Close off the previous range
            ranges.append([start_idx, prev_idx])
            # Start a new range
            start_idx = i
        prev_idx = i

    # Close off the last range
    ranges.append([start_idx, prev_idx])

    return np.array(ranges, dtype=int)


def get_burst_ranges_cell(global_zs, bs_ranges):
    """
    Creates a list of burst ranges for each burst-suppression (BS) episode.

    Parameters
    ----------
    global_zs : np.ndarray
        1D array of length `num_samples`. 0 indicates suppression, 1 indicates burst.
    bs_ranges : np.ndarray
        2D array of shape (num_bs_episodes, 2). Each row is [start_index, end_index]
        for a burst-suppression episode.

    Returns
    -------
    burst_ranges_cell : list of np.ndarray
        A list where each element corresponds to a BS episode. 
        Each element is a 2D array of shape (num_bursts, 2) specifying start and end
        indices for bursts in that episode.
    """
    # Indices where global_zs == 1
    all_burst_indices = np.where(global_zs == 1)[0]

    num_bs = bs_ranges.shape[0]  # number of BS episodes
    burst_ranges_cell = []

    for k in range(num_bs):
        start_ind = bs_ranges[k, 0]
        end_ind   = bs_ranges[k, 1]

        # Get all bursts within the [start_ind, end_ind] interval
        mask = (all_burst_indices >= start_ind) & (all_burst_indices <= end_ind)
        burst_indices = all_burst_indices[mask]

        # Convert consecutive burst indices to [start, end] ranges
        burst_ranges = convert_indices_to_index_ranges(burst_indices, min_gap=2)

        burst_ranges_cell.append(burst_ranges)

    return burst_ranges_cell


def label_global_zs(data, srate, suppression_threshold=None):
    """
    Label each sample as burst (1) or suppression (0).

    Parameters
    ----------
    data : np.ndarray
        Shape (num_channels, num_samples)
    srate : float
        Sampling rate
    suppression_threshold : float, optional
        Custom threshold for suppression detection. If None, uses adaptive threshold.

    Returns
    -------
    global_zs : np.ndarray
        1D array of length num_samples (0 => suppression, 1 => burst).
    """
    num_chans, num_samps = data.shape
    global_zs = np.zeros(num_samps, dtype=int)

    if suppression_threshold is None:
        # Calculate adaptive threshold based on data
        # Use first 60 seconds to estimate threshold
        sample_size = min(int(60 * srate), num_samps)
        sample_data = data[:, :sample_size]
        amplitudes = np.array([np.mean(np.abs(sample_data[:, i])) for i in range(sample_size)])
        # Use 15th percentile as suppression threshold (clinically reasonable)
        suppression_threshold = np.percentile(amplitudes, 15)
        print(f"   Using adaptive suppression threshold: {suppression_threshold:.2f} μV")
    else:
        print(f"   Using fixed suppression threshold: {suppression_threshold:.2f} μV")
    
    for i in range(num_samps):
        # Simple approach: mean absolute amplitude across channels
        val = np.mean(np.abs(data[:, i]))
        if val > suppression_threshold:
            global_zs[i] = 1  # classify as "burst" (above threshold)
        else:
            global_zs[i] = 0  # classify as "suppression" (below threshold)

    return global_zs


def calculate_bsr(global_zs, srate, is_artifact):
    """
    Calculate the burst-suppression ratio (BSR) at each sample.

    Parameters
    ----------
    global_zs : np.ndarray
        1D array of length `num_samples` (0 => suppression, 1 => burst).
    srate : float
        Sampling rate (Hz).
    is_artifact : np.ndarray
        1D binary array of length `num_samples`, where 1 => artifact, 0 => valid.
    
    Returns
    -------
    bsr : np.ndarray
        1D array of length `num_samples`, giving the local fraction of suppression
        (global_zs == 0) around each sample (excluding artifacts).
    """
    # Use a 60-second window for BSR calculation (standard in clinical practice)
    window_duration = 60.0  # seconds - clinically standard BSR window
    window_size = int(np.round(window_duration * srate))

    num_samps = len(global_zs)
    bsr = np.zeros(num_samps, dtype=float)

    for i in range(num_samps):
        start_i = max(0, i - window_size // 2)
        end_i = min(num_samps, i + window_size // 2)

        # Exclude artifact
        valid_mask = (is_artifact[start_i:end_i] == 0)
        local_zs = global_zs[start_i:end_i][valid_mask]

        if len(local_zs) == 0:
            bsr[i] = 0.0
        else:
            # fraction of samples that are "suppression" (== 0)
            frac_supp = np.sum(local_zs == 0) / len(local_zs)
            bsr[i] = frac_supp

    return bsr


def calculate_bs_index_ranges(bsr, srate):
    """
    Identify contiguous index ranges where the BSR indicates 
    burst-suppression (above some threshold).

    Parameters
    ----------
    bsr : np.ndarray
        1D array of floats in [0,1], of length `num_samples`.
    srate : float
        Sampling rate (not necessarily used if your threshold is direct)

    Returns
    -------
    bs_ranges : np.ndarray
        2D array of shape (N, 2) with [start_index, end_index] for each
        contiguous region identified as burst-suppression.
    """
    threshold = 0.5  # example
    is_bs = (bsr > threshold)

    bs_indices = np.where(is_bs)[0]
    # Re-use the convert_indices_to_index_ranges function, or implement the grouping logic.
    bs_ranges = convert_indices_to_index_ranges(bs_indices, min_gap=2)

    return bs_ranges


def get_params(*args):
    """
    Return one or more DetectBsParams by name, similar to MATLAB's get_params method.
    
    Parameters
    ----------
    *args : str
        One or more parameter names (e.g., 'min_bs_time', 'forgetting_time', etc.)

    Returns
    -------
    A single value (if one parameter name is given) or a tuple of values (if multiple are given).

    Examples
    --------
    >>> min_bs_time = get_params('min_bs_time')
    >>> forgetting_time, burst_thresh = get_params('forgetting_time', 'burst_threshold')
    """
    # Dictionary of parameter names and their values
    params_dict = {
        # Params for labeling z's (burst vs suppression)
        # Local z's labeling
        'forgetting_time':       0.1047,  # controls how much of recursive mean/variance is based on past
        'burst_threshold':       1.75,    # min variance for sample to be considered a burst

        # Combining z's from local to global
        'agree_percent':         0.6,     # fraction of channels needing to agree on a '1' 
        'min_suppression_time':  0.5,     # minimum duration (secs) of a suppression

        # Params for BSR
        'bsr_low_cutoff':        0.5,
        'bsr_high_cutoff':       1.0,
        'bsr_window':            60,      # window length in seconds for smoothing used to calculate BSR

        # Params for getting BS episodes from BSR
        'min_bs_time':           30, # minimum duration (secs) of a BS episode considered (reduced for testing)
        'bs_episode_smoothing_amount': 60 # max gap (secs) between episodes to consider as continuous
    }

    # Collect requested parameters
    results = []
    for arg in args:
        if arg not in params_dict:
            raise KeyError(f"Unknown parameter '{arg}' requested. Check get_params dictionary.")
        results.append(params_dict[arg])

    # Return a single value if only one parameter was requested, otherwise a tuple
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def detect_bs(eeg):
    """
    Main burst suppression detection function.
    
    Parameters
    ----------
    eeg : dict
        Dictionary with keys:
        - 'data': np.ndarray of shape (num_channels, num_samples)
        - 'srate': float, sampling rate in Hz
    
    Returns
    -------
    bs_ranges : np.ndarray
        Array of [start_idx, end_idx] for each burst-suppression episode
    global_zs : np.ndarray
        1D array (0 => suppression, 1 => burst)
    bsr : np.ndarray
        Burst-suppression ratio array
    burst_ranges_cell : list
        List of burst ranges for each BS episode
    """
    data = eeg['data']    # shape (num_channels, num_samples)
    srate = eeg['srate']

    # Create a dummy artifact array (all zeros - no artifacts)
    is_artifact = np.zeros(data.shape[1], dtype=int)

    # 1) Identify bursts vs. suppression at each sample
    global_zs = label_global_zs(data, srate)

    # 2) Compute burst-suppression ratio
    bsr = calculate_bsr(global_zs, srate, is_artifact)

    # 3) Identify contiguous burst-suppression episodes
    bs_ranges = calculate_bs_index_ranges(bsr, srate)

    # 4) Enforce minimum BS episode length
    min_bs_time = get_params('min_bs_time') 
    min_bs_slength = int(min_bs_time * srate)

    # Filter out short episodes
    filtered_bs = []
    if bs_ranges.size > 0:  # Check if there are any ranges
        for rng in bs_ranges:
            if (rng[1] - rng[0]) >= min_bs_slength:
                filtered_bs.append(rng)
    
    if len(filtered_bs) > 0:
        bs_ranges = np.array(filtered_bs)
    else:
        bs_ranges = np.zeros((0, 2), dtype=int)

    # 5) Build the list of burst-range blocks
    burst_ranges_cell = get_burst_ranges_cell(global_zs, bs_ranges)

    return bs_ranges, global_zs, bsr, burst_ranges_cell


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_eeg_channels(eeg_data, srate, start_sample=0, end_sample=800, title="EEG Visualization"):
    """
    Plot EEG channels.
    
    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data of shape (num_channels, num_samples)
    srate : float
        Sampling rate
    start_sample : int
        Start sample index
    end_sample : int
        End sample index
    title : str
        Plot title
    """
    n_samples = end_sample - start_sample
    time = np.arange(n_samples) / srate
    n_channels = eeg_data.shape[0]

    fig, axs = plt.subplots(n_channels, 1, figsize=(12, 12), sharex=True)
    fig.subplots_adjust(hspace=0.5)

    for channel in range(n_channels):
        axs[channel].plot(
            time,
            eeg_data[channel][start_sample:end_sample]
        )
        axs[channel].set_ylabel(f'Ch {channel + 1} (μV)', fontsize=10)
        axs[channel].tick_params(axis='both', which='major', labelsize=8)

    axs[-1].set_xlabel('Time (Seconds)', fontsize=12)
    fig.suptitle(title, fontsize=16)
    plt.show()


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def analyze_eeg_amplitudes(data, srate, duration_sec=10):
    """
    Analyze EEG amplitude distribution to help set appropriate thresholds.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (num_channels, num_samples)
    srate : float
        Sampling rate
    duration_sec : float
        Duration to analyze (seconds)
    """
    num_samples = int(duration_sec * srate)
    if num_samples > data.shape[1]:
        num_samples = data.shape[1]
    
    # Calculate mean absolute amplitude across channels for each sample
    sample_data = data[:, :num_samples]
    amplitudes = np.array([np.mean(np.abs(sample_data[:, i])) for i in range(num_samples)])
    
    print(f"\n=== EEG AMPLITUDE ANALYSIS (first {duration_sec}s) ===")
    print(f"Mean amplitude: {np.mean(amplitudes):.2f} μV")
    print(f"Median amplitude: {np.median(amplitudes):.2f} μV")
    print(f"Standard deviation: {np.std(amplitudes):.2f} μV")
    print(f"Min amplitude: {np.min(amplitudes):.2f} μV")
    print(f"Max amplitude: {np.max(amplitudes):.2f} μV")
    print(f"5th percentile: {np.percentile(amplitudes, 5):.2f} μV")
    print(f"95th percentile: {np.percentile(amplitudes, 95):.2f} μV")
    
    # Suggest appropriate thresholds
    suggested_threshold = np.percentile(amplitudes, 20)  # 20th percentile as suppression threshold
    print(f"\nSuggested suppression threshold (20th percentile): {suggested_threshold:.2f} μV")
    
    return amplitudes, suggested_threshold


def run_burst_detection_pipeline(filename="0284_005_008_EEG.mat"):
    """
    Run the complete EEG burst suppression detection pipeline.
    
    Parameters
    ----------
    filename : str
        Name of the EEG file to process
        
    Returns
    -------
    results : dict
        Dictionary containing all analysis results
    """
    print("=== EEG BURST SUPPRESSION DETECTION PIPELINE ===\n")
    
    try:
        # Step 1: Extract header information
        print("1. Extracting header information...")
        fs, channels = extract_header("0284_005_008_EEG")
        fs_target = 200
        print(f"   Original sampling rate: {fs} Hz")
        print(f"   Target sampling rate: {fs_target} Hz")
        print(f"   Number of channels: {len(channels)}")
        print(f"   Channel names: {channels}")
        
        # Step 2: Load EEG data
        print(f"\n2. Loading EEG data from {filename}...")
        loaded_eeg, channel_mapping = load_mat_eeg(fs, channels, filename)
        print(f"   Loaded EEG shape: {loaded_eeg.shape}")
        print(f"   Data type: {loaded_eeg.dtype}")
        
        # Step 3: Preprocess EEG data
        print("\n3. Preprocessing EEG data...")
        preprocessed_eeg = preprocess(loaded_eeg, fs_target)
        print(f"   Preprocessed EEG shape: {preprocessed_eeg.shape}")
        print(f"   Sample values (first 5): {preprocessed_eeg[0, :5]}")
        
        # Step 3.5: Analyze EEG amplitudes to set appropriate threshold
        amplitudes, suggested_threshold = analyze_eeg_amplitudes(preprocessed_eeg, fs_target, duration_sec=60)
        
        # Step 4: Run burst suppression detection
        print("\n4. Running burst suppression detection...")
        eeg_dict = {
            'data': preprocessed_eeg,
            'srate': fs_target
        }
        
        bs_ranges, global_zs, bsr, burst_ranges_cell = detect_bs(eeg_dict)
        
        # Step 5: Analyze results
        print("\n5. Analyzing results...")
        suppression_count = np.sum(global_zs == 0)
        burst_count = np.sum(global_zs == 1)
        total_samples = len(global_zs)
        
        print(f"\n=== DETECTION RESULTS ===")
        print(f"Total samples: {total_samples}")
        print(f"Suppression samples: {suppression_count} ({suppression_count/total_samples*100:.1f}%)")
        print(f"Burst samples: {burst_count} ({burst_count/total_samples*100:.1f}%)")
        print(f"Mean BSR: {np.mean(bsr):.3f}")
        print(f"BS episodes detected: {len(bs_ranges)}")
        
        if len(bs_ranges) > 0:
            print("\nBS episodes (sample indices):")
            for i, rng in enumerate(bs_ranges):
                duration_sec = (rng[1] - rng[0]) / fs_target
                print(f"   Episode {i+1}: samples {rng[0]}-{rng[1]} (duration: {duration_sec:.1f}s)")
        
        print("\n✅ SUCCESS: Burst suppression detection completed successfully!")
        
        # Return results
        results = {
            'original_fs': fs,
            'target_fs': fs_target,
            'channels': channels,
            'channel_mapping': channel_mapping,
            'loaded_eeg': loaded_eeg,
            'preprocessed_eeg': preprocessed_eeg,
            'bs_ranges': bs_ranges,
            'global_zs': global_zs,
            'bsr': bsr,
            'burst_ranges_cell': burst_ranges_cell,
            'suppression_count': suppression_count,
            'burst_count': burst_count,
            'total_samples': total_samples
        }
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete pipeline
    results = run_burst_detection_pipeline()
    
    if results is not None:
        print(f"\n=== PIPELINE COMPLETED ===")
        print(f"Results available in 'results' dictionary")
        print(f"Key results:")
        print(f"- BS episodes found: {len(results['bs_ranges'])}")
        print(f"- Suppression percentage: {results['suppression_count']/results['total_samples']*100:.1f}%")
        print(f"- Mean BSR: {np.mean(results['bsr']):.3f}")
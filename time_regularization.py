import numpy as np

def regularize_timestamps(data, fs, gap_thresh=0.25):
    """
    Parameters
    ----------
    data  : 2-D array, shape (N, C+1)
        Samples with original timestamps (e.g., from PLL clock) in the first column, where N is the
        number of samples and C is the number of channels.
    fs : float
        Nominal sampling rate (Hz).
    gap_thresh : float
        Any inter-sample gap larger than this (seconds) is treated
        as a true interruption and filled with NaNs.

    Returns
    -------
    ts_uniform : 1-D array
        New uniformly-spaced timeline.
    data_uniform : 2-D array
        Data re-indexed onto `ts_uniform`, NaNs in gaps.
    """
    ts_raw   = data[:, 0].copy()  # first column is timestamps
    data = data[:, 1:]  # remaining columns are data channels
    
    dt_nom  = np.float64(1.0 / fs)  # nominal sampling interval in seconds
    # 1. find gap boundaries
    gaps = np.where(np.diff(ts_raw) > gap_thresh)[0] + 1
    seg_starts = np.r_[0, gaps]
    seg_ends   = np.r_[gaps, len(ts_raw)]

    out_t   = []
    out_dat = []

    for s0, s1 in zip(seg_starts, seg_ends):
        seg_len = s1 - s0
        if seg_len == 0:
            continue
        t0    = ts_raw[s0]
        t_seg = t0 + np.arange(seg_len) * dt_nom
        out_t.append(t_seg)
        out_dat.append(data[s0:s1])

        # pad the gap (except after last segment)
        if s1 < len(ts_raw):
            gap_dur    = ts_raw[s1] - t_seg[-1]          # actual gap
            n_pad      = int(np.round(gap_dur * fs)) - 1 # already have 1 dt
            if n_pad > 0:
                pad_t   = t_seg[-1] + dt_nom + np.arange(n_pad) * dt_nom
                out_t.append(pad_t)
                out_dat.append(np.full((n_pad, data.shape[1]), np.nan))

    ts_uniform   = np.concatenate(out_t)
    data_uniform = np.vstack(out_dat)
    
    # assert ts is uniformly spaced:
    if not np.allclose(np.diff(ts_uniform), dt_nom, rtol=1e-6, atol=1e-8):
        raise ValueError("Timestamps are not uniformly spaced.")

    # We have asserted timestamps are correctly uniformly spaced, let's ensure they are exactly/precisely spaced:
    ts_uniform = ts_uniform[0] + np.arange(len(ts_uniform)) * dt_nom
    
    # concat timestamps and data
    if len(ts_uniform) != len(data_uniform):
        raise ValueError("Mismatch in lengths of timestamps and data arrays after regularization.")
    data_uniform = np.column_stack((ts_uniform, data_uniform))
    
    return data_uniform


# old snippet to check raw timestamps:

""""
if 0: # check the "raw" timestamps

    t_eeg = eeg[:, 0]
    dt_eeg = np.diff(t_eeg)

    # 2x1 subplot with t_eeg and dt_eeg
    fig, axs = plt.subplots(2, 1, figsize=(7, 3), sharex=True)
    axs[0].plot(t_eeg, label='Time (s)')
    axs[0].set_title('EEG Time')
    axs[0].set_xlabel('Sample Index')
    axs[0].legend()

    axs[1].plot(dt_eeg, label='Time Difference (s)')
    axs[1].set_title('EEG Time Difference')
    axs[1].set_xlabel('Sample Index')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 2))
    plt.hist(dt_eeg, bins=100)
    plt.title('Histogram of EEG dt')
    plt.xlabel('dt (s)')
    
"""
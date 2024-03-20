import numpy as np
from scipy.signal import welch, filtfilt, butter


# compute PSD of every trial and channel
def compute_psd(data, sampling_rate, window):

    trials, channels, _ = data.shape
    data_psd = np.zeros((trials, channels, window//2+1))
    for trial in range(trials):
        for channel in range(channels):
            frequencies, psd = welch(data[trial, channel, :], fs=sampling_rate, nperseg=window)
            data_psd[trial, channel, :] = psd

    return data_psd

def butter_lowpass_filter(data, sampling_rate, cutoff_frequency=40, order=5):

    trials, channels, _ = data.shape
    cutoff_frequency = 40
    b, a = butter(N=order, Wn=cutoff_frequency / (0.5 * sampling_rate), btype='low', analog=False)

    # Apply filter to each trial and channel
    filtered_data = np.zeros_like(data)
    for trial in range(trials):
        for channel in range(channels):
            filtered_data[trial, channel, :] = filtfilt(b, a, data[trial, channel, :])

    return filtered_data

# extract stft for each trial and channel. output dimension 65 x 64. data lowpass filtered to 40 Hz before extracting STFT
def extract_stft_features(dataset, fs, nperseg=64):
    
    trials, channels, time_series_length = dataset.shape
    # stft_features = np.zeros((trials, channels, nperseg // 2 + 1, nperseg//2+1), dtype=np.complex128)
    stft_features = np.zeros((trials, channels, nperseg // 2 + 1, nperseg//2+1))

    for trial in range(trials):
        for channel in range(channels):
            _, _, Zxx = stft(dataset[trial, channel, :].squeeze(), fs=fs, nperseg=nperseg, noverlap = 40)
            # print(Zxx.shape)
            # print(dataset[trial, channel, :].shape)
            stft_features[trial, channel, :, :] = 10*np.log10(abs(Zxx))

    return stft_features
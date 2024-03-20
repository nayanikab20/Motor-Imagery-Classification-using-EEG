

function [features] = extract_features(dataset)
%Extract features: extract power from each trials and channel 
% in 2 Hz bins with 50 percent overlap from 8 - 30 Hz.
% input: rdataset: [3xn] 3 channels, n = length of signal
% output: normalized_data = [3x11] 3 channels, n = length of signal


fs = 250;
low_freq = 8;   
high_freq = 30; 
windowSize = 2; 
overlap = 0.5 * windowSize;

% Calculate log-BP features
N = size(dataset,3);
num_trials = size(dataset,1);
num_channels = size(dataset,2);
frequencies = (0:(N-1)) * fs / N;

% power
fft_result = fft(dataset, [], 3);
power_spectrum = abs(fft_result(:,:,1:N/2 + 1)).^2 / N;

% frequency bins
frequencyBins = low_freq+overlap:windowSize:high_freq;

features = zeros(num_trials, num_channels, length(frequencyBins));

% Binning
for i = 1:length(frequencyBins)
    lower_freq = frequencyBins(i) - overlap;
    upper_freq = frequencyBins(i) + overlap;
    band_indices = find(frequencies >= lower_freq & frequencies < upper_freq);
    features(:,:,i) = log(sum(power_spectrum(:,:,band_indices), 3));
end

features = reshape(features, num_trials, []);

end
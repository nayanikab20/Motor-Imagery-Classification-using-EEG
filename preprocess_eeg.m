function [normalized_data] = preprocess_eeg(data)
% preprocess_eeg.m preprocesses eeg data.
%   removing eog artifact, notch filtering, high pass filtering, band pass filtering, removal of artifacts,
%   normalization
% input: raw: [nx3] 3 channels, n = length of signal
% output: normalized_data = [nx3] 3 channels, n = length of signal


% sampling frequency
fs = 250;

raw = data(:,1:3);
eog = data(:,4:6);

% % converting monopolar EOG to bipolar EOG
% eog_bipolar = [eog(:, 1) - eog(:,2), eog(:,3) - eog(:,2)];

% Fit regression model to remove eye blink artifact
% EEG = EOG*B; EEG=Y; EOG = X; B
Y = raw;
X = eog;
B = inv(X'* X) * X' * Y;

Y_eog_corrected = Y - X*B;

% Notch filter at 50 Hz to remove power line interference
notch_frequency = 50;
[b_notch, a_notch] = butter(5, [(notch_frequency -1), (notch_frequency +1)] / (fs/2), 'stop');
data_notch = filtfilt(b_notch, a_notch, Y_eog_corrected);

% High-pass filter with a cutoff frequency of 0.5 Hz
highpass_cutoff = 0.5;
[b_hp, a_hp] = butter(5, highpass_cutoff / (fs/2), 'high');
data_highpass = filtfilt(b_hp, a_hp, data_notch);

% Band-pass filter between 2 and 60 Hz
bandpass_low_cutoff = 2;
bandpass_high_cutoff = 60;
[b_bp, a_bp] = butter(5, [bandpass_low_cutoff, bandpass_high_cutoff] / (fs/2), 'bandpass');
data_bandpass = filtfilt(b_bp, a_bp, data_highpass);

% Clip EEG data to μ(xi) ±6σ(xi)
mean_data = mean(data_bandpass, 1);
std_data = std(data_bandpass,1);
lower_bound = mean_data - 6*std_data;
upper_bound = mean_data + 6*std_data;

clipped_data = data_bandpass;
for i = 1:size(clipped_data,2)
    clipped_data(clipped_data(:,i) < lower_bound(i),i) = lower_bound(i);
    clipped_data(clipped_data(:,i) > upper_bound(i),i) = upper_bound(i);
end

% Normalize each channel by subtracting mean and dividing by standard deviation
normalized_data = zeros(size(clipped_data));

for i = 1:size(clipped_data, 2)
    mean_channel = mean(clipped_data(:, i));
    std_channel = std(clipped_data(:, i));
    normalized_data(:, i) = (clipped_data(:, i) - mean_channel) / std_channel;
end
end
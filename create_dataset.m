function [all_trials, all_labels, all_trials_ts] = create_dataset(filename)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

data = load(filename);
data = data.data;
fs = 250;

normalized_data = cell(1,size(data,2));

% preprocessing all the data
for i = 1:size(data,2)
    normalized_data(1,i) = {preprocess_eeg(data{1,i}.X)};
end

% split data into trials
trials = cell(1,size(data,2));
labels = cell(1, size(data,2));

for j =1:size(normalized_data, 2)

    trial_start = data{1,j}.trial;
    num_trials = length(trial_start) - sum(data{1,j}.artifacts);
    length_of_trial = 7-4;
    num_channels = 3;
    dataset = zeros(num_trials, num_channels, fs*length_of_trial);
    classes = zeros(num_trials,1);
    
    i=1;
    ind=1;
    while i<=length(trial_start)
        if data{1,j}.artifacts(i) ~= true
            imagery_start = trial_start(i) + 4*fs;
            dataset(ind,:,:) = normalized_data{1,j}(imagery_start: imagery_start + fs*length_of_trial - 1,:)';
            classes(ind) = data{1,j}.y(i);
            ind = ind+1;
        else
            disp(i)
        end
        i = i+1;
    end

    trials(1,j) = {dataset};
    labels(1,j) = {classes};

end

% extract features
all_trials = [];
all_trials_ts = [];
all_labels = [];

for i = 1:size(trials,2)
    all_trials = [all_trials; extract_features(trials{1,i})];
    all_labels = [all_labels; labels{1,i}];
    all_trials_ts = [all_trials_ts; cell2mat(trials(1,i))];
end


% Convert labels 1 and 2 to 0 and 1
uniqueLabels = unique(all_labels);
all_labels = ismember(all_labels, uniqueLabels(2));  % Convert 2 to 1, leave 1 as 0

end

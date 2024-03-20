[X_train, Y_train, X_timeseries_train] = create_dataset("B04T.mat");
[X_test, Y_test, X_timeseries_test] = create_dataset("B04E.mat");

% figure,
% plot(squeeze(X_timeseries_train(10,:,:))')

save("B04E_processed.mat", "X_test","Y_test", "X_timeseries_test")
save("B04T_processed.mat", "X_train","Y_train", "X_timeseries_train")

%% testing the correctness of data
rng(98); % for reproducibility

% LDA classifier
ldaModel = fitcdiscr(X_train, Y_train);
Y_pred = predict(ldaModel, X_test);

% performance metrics
confMat = confusionmat(Y_test, Y_pred);
accuracy = sum(diag(confMat)) / sum(confMat(:));


% logistic regression classifier
logregModel = fitglm(X_train, Y_train, 'Distribution', 'binomial', 'Link', 'logit');
Y_pred_probs = predict(logregModel, X_test);
Y_pred = round(Y_pred_probs); % Convert probabilities to binary predictions


% performance metrics
confMat = confusionmat(double(Y_test), double(Y_pred));
accuracy = sum(diag(confMat)) / sum(confMat(:));

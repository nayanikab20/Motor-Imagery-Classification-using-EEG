import torch

def train_and_validate(model, train_data_loader, eval_data_loader, criterion, optimizer, num_epochs):

    train_loss = []
    train_acc = []
    validation_loss = []
    validation_acc = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total = 0
        total_loss = 0.0
        correct = 0
        

        for inputs, labels in train_data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        average_train_loss = total_loss / len(train_data_loader)
        accuracy = correct / total
        train_loss.append(average_train_loss)
        train_acc.append(accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_train_loss:.4f},Training Accuracy: {accuracy * 100:.2f}%')

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs_val, labels_val in eval_data_loader:
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                val_loss += loss_val.item()

                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()

        average_val_loss = val_loss / len(eval_data_loader)
        val_accuracy = correct_val / total_val
        validation_loss.append(average_val_loss)
        validation_acc.append(val_accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

    return train_loss, train_acc, validation_loss, validation_acc


def test(model, test_data_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_prediction_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # print(f'output: {outputs}, predicted: {predicted}')
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_prob = outputs[:,1]

            all_predictions.append(predicted.item())
            all_prediction_probs.append(predicted_prob.item())
            all_labels.append(labels.item())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    return all_labels, all_predictions, all_prediction_probs

def compute_errors(self, val_input, val_target, batch_size):
    true_labels = []
    predicted_labels = []

    for minibatch in range(0, val_input.shape[0], batch_size):
        input_batch = val_input[minibatch:minibatch+batch_size]
        target_batch = val_target[minibatch:minibatch+batch_size]

        with torch.no_grad():
            outputs = self.forward(input_batch)
            _, predicted = torch.max(outputs, 1)

        true_labels.extend(target_batch.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    # """
    # Calcular las métricas de evaluación
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(
        true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print("Accuracy:", accuracy)
    print("Precission:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

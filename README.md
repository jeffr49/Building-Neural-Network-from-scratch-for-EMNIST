# Building-Neural-Network-from-scratch-for-EMNIST

Project: EMNIST Handwritten Character Recognition Using Neural Networks
🔗 Project Repository: [Notebook link](https://www.kaggle.com/code/ajeffreyrufus/emnist-neural-network)

## 📌 Overview
I built a deep learning model from scratch to classify handwritten characters from the EMNIST Balanced dataset. The model is a fully connected neural network with skip connections to enhance feature propagation and improve accuracy. This project was part of the EMNIST Classification Challenge on Kaggle, where I implemented backpropagation manually without using high-level deep learning frameworks.

## 🔬 Key Features & Methodologies:
 Dataset: EMNIST Balanced (47 character classes)
 ### Model Architecture:
 • Input Layer: 784 neurons (28x28 grayscale image)
 • Hidden Layer: 256 neurons with ReLU activation
 • Skip Connection: Directly connects input to hidden layer
 • Output Layer: 47 neurons with Softmax activation
Optimization: Stochastic Gradient Descent (SGD) with Adaptive Learning Rate
Loss Function: Categorical Cross-Entropy
✔ Metrics: Accuracy, Precision, Recall, F1 Score, Log Loss, Specificity, AUC-ROC
✔ Evaluation: Model trained on a combined dataset and tested on the original test set

## Results & Insights:
• Achieved high accuracy in character classification with manual backpropagation
• Implemented detailed metric tracking, including AUC-ROC for multi-class classification
• Visualized model performance over epochs using Matplotlib

## Technologies Used:
NumPy
Pandas
Matplotlib
Manual Backpropagation

## Metrics:
Test Set Metrics:
Accuracy: 90.69%
Precision: 0.9080
Recall: 0.9069
F1 Score: 0.9074
Log Loss: 0.2819
Specificity: 0.9980
AUC-ROC: 0.9982

## Takeaways:
This project deepened my understanding of neural network fundamentals, weight initialization, and manual implementation of forward and backward propagation. The use of skip connections helped improve gradient flow, enhancing training efficiency.

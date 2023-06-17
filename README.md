# Deep-learning-mModel-to-Predict-DOg-or-Cat
This deep learning model predicts cats or dogs in images. It uses MobileNet V2 architecture with pre-trained weights. The dataset is split into training and testing sets, scaled, and trained. The model achieves 82.81% train accuracy and 78.75% test accuracy. It can classify custom images as cats or dogs.
Description of Deep Learning Model for Predicting Dog or Cat:

The deep learning model described here aims to predict whether an image contains a dog or a cat. The model utilizes the MobileNet V2 architecture, which is a popular pre-trained model for image classification tasks.

The model follows the following steps:

1. Data Preprocessing: The original dataset consists of cat and dog images. First, the images are resized to a uniform size of 224x224 pixels. Then, the images are converted to the RGB format if needed. The resized images are saved in separate directories for cats and dogs.

2. Train-Test Split: The dataset is divided into training and testing sets. 80% of the data is used for training, and 20% is kept for testing. The images are further scaled by dividing the pixel values by 255 to bring them within the range of 0 to 1.

3. Model Architecture: The MobileNet V2 model is imported using TensorFlow Hub. It serves as the base model for the classification task. A dense layer with 2 output units is added on top of the base model to classify the images into cat or dog. The base model's weights are frozen (trainable=False) to retain the pre-trained knowledge.

4. Model Compilation: The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.

5. Model Training: The model is trained on the training data, where the input is the scaled images (X_train_scaled) and the labels are the corresponding cat/dog class (Y_train).

6. Model Evaluation: The trained model is evaluated on the test data (X_test_scaled, Y_test) to calculate the score and accuracy. The score represents the loss value, and the accuracy represents the accuracy of the model predictions on the test data.

7. Prediction on Custom Image: The user can input the path of an image to make predictions using the trained model. The input image is loaded, resized to 224x224 pixels, and scaled. The reshaped image is fed into the model for prediction. The predicted label is obtained by finding the class with the highest probability. If the predicted label is 0, it indicates a cat; otherwise, it indicates a dog.

The accuracy achieved by the model on the train data is reported as 82.81%, and the accuracy on the test data is reported as 78.75%. These accuracy values provide an indication of how well the model is performing in distinguishing between cats and dogs based on the given dataset.

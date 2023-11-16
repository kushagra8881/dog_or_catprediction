# dog_or_catprediction
Cat or Dog Prediction Model using CNN and TensorFlow

This project builds a Convolutional Neural Network (CNN) model using TensorFlow to classify images of cats and dogs. The model is trained on a dataset of cat and dog images and achieves an accuracy of around 97% on the test set.

Prerequisites:

    Python 3.6 or higher
    TensorFlow 2.5 or higher
    NumPy
    OpenCV (for image pre-processing)

Instructions:

    Clone the repository:

Bash

git clone https://github.com/bard/cat-or-dog-prediction-model.git

Use code with caution. Learn more

    Install the required dependencies:

Bash

pip install tensorflow numpy opencv-python

Use code with caution. Learn more

    Download the dataset:

    Download the Dogs vs. Cats: https://www.kaggle.com/c/dogs-vs-cats/overview dataset from Kaggle.
    Extract the downloaded files and place them in a directory named data.

    Train the model:

Bash

python train.py

Use code with caution. Learn more

This will train the CNN model and save the trained model weights to a file named model.h5.

    Evaluate the model:

Bash

python evaluate.py

Use code with caution. Learn more

This will evaluate the trained model on a test set of images and print the accuracy.

    Predict on new images:

Bash

python predict.py <image_path>

Use code with caution. Learn more

This will predict the class (cat or dog) of the image specified by the <image_path>.

Example Usage:
Bash

python predict.py images/cat.jpg

Use code with caution. Learn more

This will predict the class of the image images/cat.jpg. The output will be something like:

Predicted class: cat
Probability: 0.9999

This indicates that the model is 99.99% confident that the image is a cat.

**Sign Language Recognition with Convolutional Neural Network (CNN)**

This project demonstrates a method for sign language recognition using a convolutional neural network (CNN). The goal is to train a CNN model to recognize and classify the letters of the alphabet represented by sign language gestures.

**Prerequisites**

Make sure you have the following libraries installed:

tensorflow
scikit-learn
numpy
opencv-python
matplotlib
scikeras

**Image Organization**

The sign language images should be organized in subdirectories, where each subdirectory represents a letter of the alphabet. For example, the directory structure can be as follows:

DATA/

A/

image1.jpg

image2.jpg

...

B/

image1.jpg

image2.jpg

...

C/

image1.jpg

image2.jpg

...

...

Make sure the images are named correctly, preferably with the .jpg extension.

**Project Inputs**

The following inputs are required to run the project:

1. data_directory: The path to the directory that contains the sign language images.
2. param_grid: The dictionary that defines the grid of values for the hyperparameters you want to tune. Add/remove hyperparameters and their values according to your needs.

**Compilation and Execution**

1. Run the following command: [pip install -r requirements.txt]
2. Organize the sign language images into subdirectories, following the structure mentioned above.
3. Configure the project inputs in the source code, in the indicated locations.
4. Run the Python code with the following command: [python main.py]
   
**Results**

After running the project, you will receive the following results:

 - The trained CNN model with the best hyperparameters found using grid search will be displayed.
 - The performance of the model will be evaluated based on the test data, providing the loss and accuracy.
 - A visualization of the test images along with their corresponding letters will be displayed, showing the true letter and the letter predicted by the model.

# Forest_Fire_Detection_using_DEEP_LEARNING
This project uses a Convolutional Neural Network (CNN) to detect forest fires in images. It's designed to help with the crucial task of early wildfire detection and prevention by accurately and efficiently identifying the presence of fire in images.

## 🌟 Key Features

* 🔥 **Image-based fire detection:** The system analyzes images to determine the presence of fire.
* 🧠 **CNN Model:** A Convolutional Neural Network architecture is used for image classification.
* 🛠️ **Implemented with TensorFlow and Keras:** Built using popular deep learning libraries.
* ✨ **Data Preprocessing:** Images are preprocessed (resized, normalized) to improve model performance.
* 📊 **Training and Validation:** The model is trained on a dataset of fire and non-fire images, with a portion of the data used for validation.
* ✅ **Evaluation:** The model's performance is evaluated on a separate test dataset.
* 🔮 **Prediction:** The system can predict whether a new, unseen image contains fire.

## ⚙️ How it Works

The project workflow is as follows:

1.  **⬇️ Dataset Download:**

    * The code uses `kagglehub` to download the wildfire dataset from Kaggle.
    * The dataset is organized into training, validation, and testing sets, each containing images of fire and non-fire.
    * The dataset is downloaded to a local directory.

2.  **🔍 Dataset Exploration:**

    * The code explores the dataset by listing the classes (fire and non-fire) and visualizing sample images from each class.
    * This step helps to understand the nature of the images.

3.  **🧹 Data Preprocessing:**

    * Images are preprocessed to ensure they are suitable for the CNN model.
    * Image dimensions and batch size are defined.
    * `ImageDataGenerator` is used to:
        * Rescale pixel values to the range [0, 1].
        * Create data generators for the training, validation, and test sets.
        * Resize images.
        * Set the class mode to 'binary'.
        * Shuffle the training data.

4.  **🏗️ Model Building:**

    * A CNN model is built using Keras's Sequential API.
    * The model architecture consists of:
        * **Input Layer:** Defines the shape of the input images.
        * **Convolutional Layers (Conv2D):** Extract features using filters (32, 64, and 128 filters) with 'relu' activation.
        * **Max Pooling Layers (MaxPooling2D):** Reduce spatial dimensions.
        * **Flatten Layer:** Converts 2D feature maps to a 1D vector.
        * **Dense Layers (Fully Connected):** Perform classification (512 units with 'relu' and Dropout, and a final 1 unit with 'sigmoid').
    * The model is compiled with the Adam optimizer, binary cross-entropy loss, and the 'accuracy' metric.
    * The model summary is printed.

5.  **🚀 Model Training:**

    * The model is trained using the `fit` method.
    * The training and validation data generators are used.
    * The number of steps per epoch and the number of epochs are specified.
    * The training history (accuracy and loss) is stored.

6.  **📊 Model Evaluation:**

    * The trained model is evaluated on the test dataset.
    * The test loss and test accuracy are printed.

7.  **📈 Visualization:**

    * The code generates plots:
        * Training and Validation Accuracy vs. Epochs
        * Training and Validation Loss vs. Epochs
    * These plots help diagnose training and identify overfitting.

8.  **🔮 Prediction on New Images:**

    * The `predict_fire(image_path)` function predicts whether a new image contains fire.
    * The function loads, preprocesses, and passes the image to the model.
    * The model's output probability is converted to a class label (fire or no fire).
    * The input image and prediction are displayed.

## 🛠️ How to Use

1.  **Prerequisites:**

    * Python 3.x
    * TensorFlow
    * Keras
    * Matplotlib
    * Kaggle account
    * `kagglehub` library

2.  **Installation:**

    ```bash
    pip install tensorflow matplotlib kagglehub
    ```

3.  **Download the dataset:**

    * Run the Python script.

4.  **Run the code:**

    * Execute the Python script. The script will:
        * Download the dataset.
        * Preprocess the images.
        * Build and train the CNN model.
        * Evaluate the model.
        * Display the training/validation plots.
        * (Optionally) Predict on a new image.

5.  **Predicting on New Images (Optional):**

    * The python script contains a `predict_fire(image_path)` function. Call it with the image path.

## Results

The code will output:

* Path to the downloaded dataset.
* Number of classes and class names.
* CNN model summary.
* Training and validation accuracy and loss curves.
* Test loss and test accuracy.
* Predicted class of a new image (if used).

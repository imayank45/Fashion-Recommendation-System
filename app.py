# Import the TensorFlow library
import tensorflow


# Import the image preprocessing module from Keras
from tensorflow.keras.preprocessing import image

# Import the GlobalMaxPooling2D layer from Keras
from tensorflow.keras.layers import GlobalMaxPooling2D

# Import the ResNet50 model and the preprocess_input function for image preprocessing
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Import NumPy library with the alias np
import numpy as np

# Import the norm function from the NumPy linear algebra module
from numpy.linalg import norm

# Import the os module for interacting with the operating system
import os

# Import the 'tqdm' module for creating progress bars
from tqdm import tqdm

# Import the Image module from the Pillow library
from PIL import Image

# Import the 'pickle' module for working with pickled objects (serialization)
import pickle

# Create a ResNet50 model with pre-trained weights
model = ResNet50(

    # Use pre-trained ImageNet weights
    weights = 'imagenet',

    # Exclude the fully-connected layers at the top
    include_top = False,

    # Set the input shape to (224, 224, 3)
    input_shape = (224,224,3)
)


# Set the ResNet50 layers as non-trainable
model.trainable = False


# Create a new sequential model by stacking the ResNet50 model and GlobalMaxPooling2D layer
model = tensorflow.keras.Sequential([

    # Add the ResNet50 model
    model,

    # Add a GlobalMaxPooling2D layer to reduce spatial dimensions
    GlobalMaxPooling2D()
])


def extract_features(img_path, model):

    # Load and resize the image to the target size expected by the model (224x224 pixels)
    img = Image.open(img_path)
    img = img.resize((224, 224))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Check if the image has only one channel (grayscale), and if so, convert it to RGB
    if img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)

    # Add an extra dimension to the array to match the input shape expected by the model
    expanded_image_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image array to align with the preprocessing used during training
    preprocessed_img = preprocess_input(expanded_image_array)

    # Use the model to predict features for the preprocessed image and flatten the result
    result = model.predict(preprocessed_img).flatten()

    # Normalize the result by dividing by the L2 norm (Euclidean norm) of the feature vector
    normalized_result = result / norm(result)

    # Return the normalized feature vector
    return normalized_result




# List to store filenames
filenames = []

# Loop through files in the 'images' directory and append their full paths to the list
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))


# List to store extracted features
feature_list = []

# Loop through each filename and extract features using the 'extract_features' function with the 'model'
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))


# Save the 'feature_list' to the file 'embeddings.pkl'
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))

# Save the 'filenames' to the same file 'embeddings.pkl'
pickle.dump(filenames, open('filenames.pkl', 'wb'))

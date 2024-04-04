import pickle

import numpy as np

import tensorflow

from tensorflow.keras.layers import GlobalMaxPooling2D

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from PIL import Image

from numpy.linalg import norm

from sklearn.neighbors import NearestNeighbors

import cv2


feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))

filenames = pickle.load(open('filenames.pkl', 'rb'))

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


# Load and resize the image to the target size expected by the model (224x224 pixels)
img = Image.open('sample/dress.jpg')
img = img.resize((224, 224))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Add an extra dimension to the array to match the input shape expected by the model
expanded_image_array = np.expand_dims(img_array, axis=0)

# Preprocess the image array to align with the preprocessing used during training
preprocessed_img = preprocess_input(expanded_image_array)

# Use the model to predict features for the preprocessed image and flatten the result
result = model.predict(preprocessed_img).flatten()

# Normalize the result by dividing by the L2 norm (Euclidean norm) of the feature vector
normalized_result = result / norm(result)


neighbours = NearestNeighbors(
    n_neighbors=6,
    algorithm='brute',
    metric='euclidean'
)

neighbours.fit(feature_list)

distances, indices = neighbours.kneighbors([normalized_result])

print(indices)


for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)

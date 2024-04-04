# Importing streamlit to create web-applications
import streamlit as st

# Importing operating-system
import os

# Import the Image module from the Pillow library
from PIL import Image

# Import NumPy library with the alias np
import numpy as np

# Import the 'pickle' module for working with pickled objects (serialization)
import pickle

# Import the GlobalMaxPooling2D layer from Keras
from tensorflow.keras.layers import GlobalMaxPooling2D

# Import the image preprocessing module from Keras
from tensorflow.keras.preprocessing import image

# Import the ResNet50 model and the preprocess_input function for image preprocessing
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Importing the NearestNeighbors class from the scikit-learn library
from sklearn.neighbors import NearestNeighbors

# Import the TensorFlow library
import tensorflow

# Import the norm function from the NumPy linear algebra module
from numpy.linalg import norm


# Load the contents of 'embeddings.pkl' file into a NumPy array and assign it to the variable 'feature_list'
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))

# Load the contents of 'filenames.pkl' file and assign it to the variable 'filenames'
filenames = pickle.load(open('filenames.pkl', 'rb'))


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

st.title('Fashion Recommender System')


# Define a function named save_uploaded_file that takes an uploaded_file as a parameter
def save_uploaded_file(uploaded_file):
    try:

        # Open a file in binary write mode in the 'uploads' directory
        # The file name is obtained by joining the 'uploads' directory with the name of the uploaded file
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:

            # Write the contents of the uploaded file to the opened file
            f.write(uploaded_file.getbuffer())

        # If the file is successfully saved, return 1 (indicating success)
        return 1
    except:

        # If an exception (error) occurs during the process, return 0 (indicating failure)
        return 0


# Define a function to extract features from an image using a given model
def feature_extraction(img_path, model):

    # Load and resize the image to the target size expected by the model (224x224 pixels)
    img = Image.open(img_path)
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

    # Return the normalized feature vector
    return normalized_result



# Define the recommend function that takes features and a feature_list as input
def recommend(features, feature_list):
    neighbours = NearestNeighbors(

        # Create a NearestNeighbors object with parameters:
        # - n_neighbors: Number of neighbors to use (6 in this case)
        # - algorithm: Algorithm used to compute nearest neighbors ('brute' in this case)
        # - metric: Distance metric to use for the tree ('euclidean' in this case)
        n_neighbors=6,
        algorithm='brute',
        metric='euclidean'
    )

    # Fit the NearestNeighbors model with the provided feature_list
    neighbours.fit(feature_list)

    # Find the distances and indices of the 6 nearest neighbors for the given features
    distances, indices = neighbours.kneighbors([features])

    # Return the indices of the nearest neighbors
    return indices



# UI part using Streamlit
uploaded_file = st.file_uploader('Choose an image')

# Check if an image is uploaded
if uploaded_file is not None:

    # Check if the uploaded file is successfully saved
    if save_uploaded_file(uploaded_file):

        # Display the uploaded image
        display_image = Image.open(uploaded_file)

        # Extract features from the uploaded image using a model
        features = feature_extraction(os.path.join('uploads', uploaded_file.name), model)

        # Recommend images based on the extracted features
        indices = recommend(features, feature_list)

        # Display recommended images in five columns
        col1, col2= st.columns(2)

        with col1:
            st.subheader('Uploaded Image')
            st.image(display_image,use_column_width=True)

        with col2:
            st.subheader('Results ')
            x1,x2,x3,x4 = st.columns(4)
            with x1:
                st.image(filenames[indices[0][0]])
                st.image(filenames[indices[0][1]])
            with x2:
                st.image(filenames[indices[0][2]])
                st.image(filenames[indices[0][3]])
            with x3:
                st.image(filenames[indices[0][4]])

            with x4:
                pass



    else:

        # Display an error message if there's an issue with file upload
        st.header('Some error occured in file upload')

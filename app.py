import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array

st.title('Image Classification with tensorflow')
st.write('Upload an image to classify')
uploaded_file = st.file_uploader('Choose an image...', type=["jpg","jpeg","png"])
generated_pred = st.button('Predict')
model = tf.keras.models.load_model('checkpoints/model_vgg16.keras')
classes_prediction = {'glioma_tumor': 0, 'meningioma_tumor': 1, 'no_tumor': 2, 'pituitary_tumor': 3}
if uploaded_file is not None:
    st.image(uploaded_file, caption='Image Telechargee', use_container_width=True)
    test_image = image.load_img(uploaded_file, target_size=(64,64,3))
    img_array = img_to_array(test_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisation

    
    if generated_pred:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        for key,value in classes_prediction.items():
            if value == predicted_class:
                st.title(f'Predicted class: {key}')
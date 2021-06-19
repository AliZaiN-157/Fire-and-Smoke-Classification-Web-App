import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Fire Smoke Detection Web App')

original_location_of_model = "D:/work/DeepLeaning_Project/smoke_fire/model/Fire_and_Smoke_model.h5"


prediction = st.beta_container()
dataset = st.beta_container()
model = st.beta_container()

with prediction:
    st.header('Welcome to Prediction Corner')

    def main():
        upload_file = st.file_uploader(
            "Choose the Image file ...", type=["jpg", "png", "jpeg"])

        if upload_file is not None:
            predict_image = Image.open(upload_file)
            Catagories = ['Fire', 'Smoke']
            result = predict_class(predict_image)
            predict_image = predict_image.resize((256, 256), Image.NEAREST)
            caption = (Catagories[int(result[0][0])])
            st.image(predict_image, caption)

    def predict_class(image):
        CNN = load_model(
            r'../smoke_fire/model/fire_and_smoke_model.h5')
        shape = ((256, 256, 3))
        model = tf.keras.Sequential([hub.KerasLayer(CNN, input_shape=shape)])
        test_image = image.resize((256, 256))
        test_image = preprocessing.image.img_to_array(test_image)
        test_image = test_image/255.0
        test_image = np.expand_dims(test_image, axis=0)
        predictions = model.predict(test_image)
        return predictions

# with dataset:
#     st.header('Fire Smoke Image Dataset')

# with model:
#     st.header('Trained Model')

if __name__ == '__main__':
    main()

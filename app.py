import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Maize Leaf Disease Prediction App")
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache_resource
def load_model():
	model = tf.keras.models.load_model('maize-leaf-disease-model.h5')
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [256, 256])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Maize Leaf Disease Prediction')

file = st.file_uploader("Upload a maize leaf image", type=["jpg"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['Common Rust', 'Grey Leaf Spot', 'Healthy', 'Northern Leaf Blight']

	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)
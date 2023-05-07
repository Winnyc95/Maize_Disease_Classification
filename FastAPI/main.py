# Importing libraries
from fastapi import FastAPI, File, UploadFile
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

# Creating the app object
app = FastAPI()

# Loading the trained model
model = tf.keras.models.load_model('maize-leaf-disease-model.h5')

# Creating class names
class_names = ['Common Rust', 'Grey Leaf Spot', 'Healthy', 'Northern Leaf Blight']


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# Define predict function
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    predictions = model.predict(img_batch)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
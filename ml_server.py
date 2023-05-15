import json
import tensorflow as tf
import numpy as np
import random

from flask import Flask, request

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
feature_model = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers]
)

_, (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0

def get_prediction(image):
    image_arr = np.reshape(image, (1, 784))
    return feature_model.predict(image_arr), image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the drawn image from the request
        image = np.array(request.json['image'])

        # Process the drawn image and get predictions
        preds, processed_image = get_prediction(image)
        final_preds = [p.tolist() for p in preds]

        return json.dumps({
            'prediction': final_preds,
            'image': processed_image.tolist()
        })
    return 'Welcome to the model server!'

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

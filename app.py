import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

URI = 'http://127.0.0.1:5000'
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Neural Network Visualizer')
st.sidebar.markdown('## Input Image')

# Specify canvas parameters in application
#drawing_mode = st.sidebar.selectbox(
#    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
#)

#stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
#if drawing_mode == 'point':
#    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
#stroke_color = st.sidebar.color_picker("Stroke color hex: ")
#bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
#bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)



# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width='20',
    stroke_color='#000000',
    background_color='#EEEEEE',
    #background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=300,
    width=150,
    drawing_mode='freedraw',
    #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)


if st.button('Get Prediction'):
    # Convert the drawn image to grayscale and invert the colors
    image = canvas_result.image_data.astype(np.uint8)
    image = Image.fromarray(image).convert('L')
    image = Image.fromarray(255 - np.array(image))
    image = image.resize((28, 28))

    # Convert the image to a numpy array and normalize it
    image_array = np.array(image) / 255.0

    # Send the image to the Flask server for prediction
    payload = {'image': image_array.tolist()}
    response = requests.post(URI, json=payload)
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28, 28))

    st.sidebar.image(image, width=150)

    for layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))
        num_numbers = numbers.shape[0]  # Get the number of elements in the `numbers` array
        plt.figure(figsize=(32, 4))

        if layer == 2:
            row = 1
            col = 10
        else:
            row = 2
            col = 16

        # Adjust the number of subplots based on the number of elements
        num_subplots = min(num_numbers, row * col)

        for i, number in enumerate(numbers[:num_subplots]):
            plt.subplot(row, col, i + 1)
            plt.imshow(number * np.ones((8, 8, 3)).astype('float32'))
            plt.xticks([])
            plt.yticks([])

            if layer == 2:
                plt.xlabel(str(i), fontsize=40)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        st.text('Layer {}'.format(layer + 1))
        st.pyplot()
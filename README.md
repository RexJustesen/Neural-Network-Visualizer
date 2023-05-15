# Neural-Network-Visualizer
Visualize Neural Network layers using Streamlit

# Create Model
the file called mnist_model.py Creates a Neural Network using
the mnist data set. It then saves the model file for this Neural Network. The model uses images of hand
drawn numbers and infers what the number that is written.

#Create Flask server backend
The file called ml_server.py is a Flask server that is the backend for the streamlit web application. it contains
the model files layers and the test set of the mnist data set to grab random images and send the 
layers and image data in a json packet to the webapp.

#Streamlit App
The file called app.py is a streamlit web app that allows the user to let the backend pick a random image from the test date set
or hand write their own number in a canvas and submit it. The backend then returns the output of each hidden layer and the output layer
visually. This is a great way to visualize the provess of a Neural Network and its output through each hidden layer. 

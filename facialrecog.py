import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.markdown("<h1 style='color: #FFACAC'>FACE DETECTION APP</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='margin-top: 0rem; color: #F2921D'>Built by AdrianX147</h6>", unsafe_allow_html=True)

st.image('Facial Structure.jpg', caption='Built by Mosaku Habeeb', width=400)

# Create a line and a space underneath
st.markdown('<hr><hr><br>', unsafe_allow_html=True)

# Add instructions to the Streamlit app interface
if st.button('Read the usage Instructions below'):
    st.success('Hello User, these are the guidelines for the app usage')
    st.write('Press the camera button for our model to detect your face')
    st.write('Use the MinNeighbour slider to adjust how many neighbors each candidate rectangle should have to retain it')
    st.write('Use the Scaler slider to specify how much the image size is reduced at each image scale')

st.markdown('<br>', unsafe_allow_html=True)

# Set the minNeighbours and Scale Factor sliders
min_Neighbours = st.slider('Adjust Min Neighbour', 1, 10, 5)
Scale_Factor = st.slider('Adjust Scale Factor', 1.01, 3.0, 1.3)

st.markdown('<br>', unsafe_allow_html=True)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')  # Fixed: removed space

# Check if cascade loaded successfully
if face_cascade.empty():
    st.error("Error loading cascade classifier. Make sure 'haarcascade_frontalface_default .xml' exists.")
else:
    if st.button('FACE DETECT'):
        # Create placeholder for video feed
        frame_placeholder = st.empty()
        stop_button = st.button('Stop Detection')
        
        # Initialize the webcam
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            st.error("Cannot access webcam. Please check your camera permissions.")
        else:
            while not stop_button:
                ret, camera_view = camera.read()
                
                if not ret:
                    st.error("Failed to grab frame from camera")
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(camera_view, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=Scale_Factor, 
                    minNeighbors=min_Neighbours, 
                    minSize=(30, 30)
                )
                
                # Draw rectangles around detected faces
                for (x, y, width, height) in faces:
                    cv2.rectangle(camera_view, (x, y), (x + width, y + height), (0, 255, 255), 2)
                
                # Convert BGR to RGB for Streamlit display
                camera_view_rgb = cv2.cvtColor(camera_view, cv2.COLOR_BGR2RGB)
                
                # Display in Streamlit
                frame_placeholder.image(camera_view_rgb, channels="RGB", use_container_width=True)
            
            # Release the camera
            camera.release()
            st.success("Camera stopped successfully")
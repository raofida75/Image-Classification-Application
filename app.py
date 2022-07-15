from tensorflow.keras.models import load_model
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.title('Traffic Sign Classification App')
st.write('This is a simple application which can classify traffic signs.')
st.write('')

def process_image(image):
    # resize the image to 32,32,3
    image = cv2.resize(image, (32,32))
    # converting shape from 32,32,3 to 1,32,32,3
    image = image[np.newaxis, ...]
    # normalize the image b/w 0 and 1
    image = image / 255
    return image

model=load_model('models/cnn_model.h5')


def predict(image):
    # process the image using the above function
    processed_image = process_image(image)
    # read the model
    
    # predicting the probabilities
    pred_prob = model.predict(processed_image)
    # index for the max probability
    prob = np.max(pred_prob)
    pred = np.argmax(pred_prob)
    # reading names of the signs
    sign_names = pd.read_csv('data/signnames.csv')
    # returning the corresponding sign name
    label = sign_names.iloc[pred].values[1]
    return prob, label

    
try: 
    # upload image
    file = st.sidebar.file_uploader("Please upload an image.", type=["jpg", "png"])
    image_disp = Image.open(file)
    # display the image
    st.image(image_disp, use_column_width=False, width=500)
    # converting image into an array
    image = np.asarray(image_disp)
    if st.button('Classify Image'):
        # predicting the output
        prob,pred = predict(image)
        prob = np.round(prob*100,3)
        st.sidebar.markdown('### Algorithm Predicts:')
        st.sidebar.success(f"It is a '{pred}' sign.")
        st.sidebar.markdown('### Probability:')
        st.sidebar.write(f'{prob}%')
    st.sidebar.warning('NOTE: Remove the current image if you want to make predictions for another image.')


except:
    print('Please upload the file')


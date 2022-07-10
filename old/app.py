# 01. Public Library
import os
import io
# 02. Second Source Library
import streamlit as st
import cv2
import numpy as np
from PIL import Image


# 03. Tomomi Research Library
from image_processing import fourier
# 03. Tomomi Research Library


FF = fourier.FF7()

#streamlit setting
st.set_page_config(layout="wide",
                   page_title="Tomomi Research, Inc.",
                   page_icon="random",
                   initial_sidebar_state="expanded",
                   menu_items={
                       'Get Help': 'https://www.tomomi-research.com',
                   })

# title
st.title('FFT and HPF Demo of Images')
st.write('')

#sidebar
st.sidebar.image('./image/Logo_Small_new.png')
st.sidebar.markdown(
    '''
    [Homepage](https://www.tomomi-research.com "Tomomi Research, Inc. Home")
    '''
)
st.sidebar.header('Control Panel')

# ---------------------------------------------
# 01. Program
# ---------------------------------------------
# '''
# 01. upload the single image
# 02. show the uploaded image
# 03. setting the HFP setting with slider
# 04. Button to accomplish the FFT and iFFT
# 05. Show the Result with grid
# '''

# 01. upload the single image
uploaded_file = st.sidebar.file_uploader('Choose a image file')
# 02. show the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    print(len(img_array.shape))
    if len(img_array.shape) >= 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_array
    st.image(
        image, caption='upload images',
        use_column_width=True
    )
    st.image(
        img_gray, caption='upload images (gray_scale)',
        use_column_width=True
    )

# 03. setting the HFP setting with slider
filter_freq = st.sidebar.slider('Please select the HPF range (from 0 to 100 %) ', min_value=0, max_value=100, step=1, value= 10)

# 04. Button to accomplish the FFT and iFFT
if st.sidebar.button('2. Run the Image FFT and iFFT'):
    img_fft, img_fmask, img_ifft = FF.FFT_HPF_filter(img=img_gray,freq= 0.01 * filter_freq)
    #image
    # st.header('Image')
    # st.image(img_array, caption='Image',use_column_width=True)
    #FFT
    st.header('FFT of Image')
    st.image((255 * FF._min_max(img_fft)).astype(np.uint8), caption='FFT of Image',use_column_width=True)

    # FFT + MASK
    st.header('FFT + Mask')
    st.image((255 * FF._min_max(img_fmask)).astype(np.uint8), caption='FFT + Mask',use_column_width=True)

    # IFFT
    st.header('FFT of Image')
    st.image(img_ifft.astype(np.uint8), caption='IFFT of Image',use_column_width=True)

    # cv2.imshow('Input Image', img)
    # cv2.imshow('FFT of Image', (255 * _min_max(img_fft)).astype(np.uint8))
    # cv2.imshow('FFT + Mask', (255 * _min_max(img_fmask)).astype(np.uint8))
    # cv2.imshow('After inverse FFT', img_ifft.astype(np.uint8))



import os
import glob
import shutil
import base64
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import inference
from itertools import cycle


# pandas display options
pd.set_option('display.max_colwidth', None)

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.jpeg')

### Function to save the uploaded files:
def save_uploaded_file(uploadedfile):
    try:
        shutil.rmtree("./tempDir")
    except Exception:
        pass
    try:
        os.makedirs("./tempDir")
        os.makedirs("./tempDir/output")
    except Exception:
        pass
    with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
        #st.balloons()
    return st.success("Saved file : {} in tempDir folder".format(uploadedfile.name))

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

def side_show():
    """Shows the sidebar components for the Image Thresholding and returns user inputs as dict."""
    inputs = {}
    with st.sidebar:
        st.write("#### Threshold for HPE")
        inputs["Threshold"] = st.number_input(
        "Threshold",min_value = 0.0, max_value = 1.0, value = 0.3, step = 0.1,
        )
    return inputs

def main():
    title = '<p style="background-color:rgb(200,100,100);color:rgb(255,255,255);text-align:center;font-size:30px;padding:10px 10px;font-weight:bold"> Computer Vision Class Capstone Project <p/>'
    st.markdown(title, unsafe_allow_html=True)
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        hpe = '<div style="border:2px solid #030833; border-radius:10px;  background:#ffffff;"><div style="padding:10px 10px 10px 0px; margin:2px; border-radius:10px; background:#030833; text-align:center;"> <span style="font-family:sans-serif; font-size:30px; color:#fff;"> 🔥 Monocular Human Pose Estimation</span></div></div>'
        st.markdown(hpe, unsafe_allow_html=True)

        hello = '<p style="font-family:Courier; color:Black; font-size: 20px;"><b><br>Hello, World! &#x1F981;<b></p>'
        st.markdown(hello, unsafe_allow_html=True)

        notion = '<p style="font-family:Courier; color:Black; font-size: 15px;"><i>📌 Please upload a single person Image, with center body and reduce the THRESHOLD if all the 16 key-points did not come.</p>'
        st.markdown(notion, unsafe_allow_html=True)

        params = side_show()
        uploadFile = st.file_uploader("Upload File",type=['png','jpeg','jpg'])
        if st.button("Process"):
            # Checking the Format of the page
            if uploadFile is not None:
                file_details = {"Filename":uploadFile.name,"FileType":uploadFile.type,"FileSize":uploadFile.size}
                st.markdown(file_details, unsafe_allow_html=True)
                img = load_image(uploadFile)
                success = '<p style="font-family:Courier; color:Black; font-size: 20px;">Image Uploaded Successfully</p>'
                st.markdown(success, unsafe_allow_html=True)
                st.balloons()
                st.image(img)
                save_uploaded_file(uploadFile)
                img_path = os.getcwd() + "/tempDir/" + uploadFile.name

                st.write("##### Given Threshold for Human Pose Estimation : ", round(params['Threshold'],2))
                hpe_img = inference.hpe_onnx_inference(img_path,params['Threshold'])                
                st.image(hpe_img, caption='Human Pose Estimation for Given Image')

                st.write("That's Bottom Up HPE approach! we never detected a bounding box for the image body, just the 16 keypoints.")
                # #st.image(images, use_column_width=True, caption=["Binary Thresholding Image","Grey Scaled Image"])
                    
            else:
                #st.write("Please Upload the Image and make sure your image is in JPG/PNG Format.")
                failed = '<p style="font-family:Courier; color:Black; font-size: 20px;">"Please Upload the Image and make sure your image is in JPG/PNG Format."</p>'
                st.markdown(failed, unsafe_allow_html=True)
    else:
        with st.sidebar:
            title = '<p style="font-family:Courier; color:Green; font-size: 18px;"> HPE : Pose Estimation is predicting the body part or joint positions of a person from an image or a video. </p>'
            st.markdown(title, unsafe_allow_html=True)

        hpe = '<div style="border:2px solid #030833; border-radius:10px;  background:#ffffff;"><div style="padding:10px 10px 10px 0px; margin:2px; border-radius:10px; background:#030833; text-align:center;"> <span style="font-family:sans-serif; font-size:30px; color:#fff;"> 🔥 Monocular Human Pose Estimation</span></div></div>'
        st.markdown(hpe, unsafe_allow_html=True)

        sentence = '<p style="font-family:Courier; color:Black; font-size: 18px;"> <br> <b>Human Pose Estimation (HPE)</b> — an image processing task which finds the configuration of a subjects joints and body parts in an image. When tackling human pose estimation, we need to be able to detect a person in the image and estimate the configuration of his joins (or keypoints). </p>'
        st.markdown(sentence, unsafe_allow_html=True)

        sentence1 = '<p style="font-family:Courier; color:Black; font-size: 18px;"> ◉ It displays a persons orientation as a graphical representation. It is essentially a series of coordinates that can be joined to represent the pose of the person. In the skeleton, each coordinate is referred to as a part (or a joint, or a key-point). A valid connection between two parts is known as a pair (or a limb).</p>'
        st.markdown(sentence1, unsafe_allow_html=True)

        sentence2 = '<p style="font-family:Courier; color:Black; font-size: 18px;">The most used backbone network for extracting image features is ResNet. Simple Baselines for Human Pose Estimation and Tracking adds a few deconvolutional layers to the ResNets C5 final convolution stage as part of the approach. </p>'
        st.markdown(sentence2, unsafe_allow_html=True)

        st.image("Images/resnet50.png")

        sentence3 = '<p style="font-family:Courier; color:Black; font-size: 18px;"> They chose this structure because the state-of-the-art Mask R-CNN uses it and because it is arguably the simplest way to produce heatmaps from deep and low resolution information.</p>'
        st.markdown(sentence3, unsafe_allow_html=True)

        st.image("Images/simple_pose.png")

        sentence4 = '<p style="font-family:Courier; color:Black; font-size: 18px;"> <i>◉ The Bottom Up Approach will be used in our case, which involves identifying the body parts (joints, limbs, or little template patches) and then combining them to create our human body. </i></p>'
        st.markdown(sentence4, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
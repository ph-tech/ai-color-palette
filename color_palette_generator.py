import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

st.title("🎨 AI Color Palette Generator")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize image for faster processing
    resized_img = cv2.resize(image, (150, 150))
    flat_img = resized_img.reshape((-1, 3))

    # Extract dominant colors using KMeans
    k = st.slider("Select number of colors", 1, 10, 5)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(flat_img)

    colors = kmeans.cluster_centers_.astype(int)

    st.subheader("🎨 Dominant Colors")

    # Show color blocks
    for i, color in enumerate(colors):
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        st.markdown(
            f'<div style="width:100px;height:50px;background-color:{hex_color};border:1px solid #000;margin-bottom:5px;"></div> {hex_color}',
            unsafe_allow_html=True,
        )

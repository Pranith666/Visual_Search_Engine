import streamlit as st
import numpy as np
import faiss
from PIL import Image
import os
import google.generativeai as genai
import re
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
import pandas as pd
import requests
from io import BytesIO
from googleapiclient.discovery import build
import io

# --- Configurations ---
imgs_path = "images/"
imgs_model_width, imgs_model_height = 224, 224
nb_closest_images = 5
genai.configure(api_key="YOUR API KEY")  # Replace with your API key.
model = genai.GenerativeModel('gemini-2.0-flash')

# Configure Google Custom Search API (replace with your keys)
API_KEY = "YOUR API KEY"  # Replace with your API key.
CSE_ID = "YOUR API KEY"  # Replace with your CSE ID.

# --- Load Data ---
@st.cache_resource
def load_data():
    train_features = np.load("pre/train_features (2).npy")
    train_files = np.load("pre/train_files.npy").tolist()
    index = faiss.read_index("pre/index.idx")
    df = pd.read_csv("styles.csv", on_bad_lines='skip')
    df['image_path'] = imgs_path + df['id'].astype(str) + ".jpg"
    train_files_local = [os.path.join(imgs_path, os.path.basename(file)) for file in train_files]
    return train_features, train_files_local, index, df

train_features, train_files, index, df = load_data()

# --- Feature Extraction (for uploaded image) ---
base_model = ResNet50(weights='imagenet', include_top=False)
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

def extract_features_single(img_path):
    img = load_img(img_path, target_size=(imgs_model_width, imgs_model_height))
    img_array = img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    return feature_extractor.predict(img_array)

# --- Retrieve Similar Images ---
def retrieve_similar_images(query_img_path):
    query_features = extract_features_single(query_img_path)
    distances, indices = index.search(query_features, nb_closest_images + 1)
    return [train_files[i] for i in indices[0][1:]]

# --- Display Results (First Interface) ---
def show_results(query_img_path):
    try:
        img_pil = Image.open(query_img_path)
        response = model.generate_content([
            "Describe the product in this image.",
            img_pil
        ])
        gemini_description = response.text

        product_links_response = model.generate_content([
            f"Provide links to buy similar products from Amazon, Walmart, and eBay based on this description: {gemini_description}"
        ])
        product_links = re.findall(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)', product_links_response.text)
    except Exception as e:
        gemini_description = f"Gemini description generation failed: {e}"
        product_links = []

    st.write(f"📝Description: {gemini_description}")

    if product_links:
        st.markdown("🛒 Product Links:")
        for i, link in enumerate(product_links): #using enumerate to get the index.
# Use HTML to create a link with a title (tooltip)
            st.markdown(f'<a href="{link}" title="{link}" target="_blank" rel="noopener noreferrer">Link {i + 1}</a>', unsafe_allow_html=True) #added the index to the visible link.
        #for link in product_links:
            #st.markdown(f'<a href="{link}" title="{link}" target="_blank" rel="noopener noreferrer">Click Here</a>', unsafe_allow_html=True)
            ##st.markdown(f'<a href="{link}" title="{link}" target="_blank" rel="noopener noreferrer">{link}</a>', unsafe_allow_html=True)
        # Use HTML to create a link with a title (tooltip)
            #st.markdown(f'<a href="{link}" title="{link}" target="_blank">{link}</a>', unsafe_allow_html=True)
    else:
            st.write("⚠️ No product links found.")

    st.image(query_img_path, caption="Query Image", width=200)
    similar_images = retrieve_similar_images(query_img_path)

    nb_closest_images = len(similar_images)
    cols = st.columns(min(nb_closest_images, 4))

    for i, img_path in enumerate(similar_images):
        with cols[i % len(cols)]:
            st.image(img_path, caption="Similar", width=200)
            similar_meta = df[df["image_path"].str.strip() == img_path.strip()]
            if not similar_meta.empty:
                similar_meta = similar_meta.iloc[0]
                item_name = similar_meta['productDisplayName']
                category = similar_meta['masterCategory']
                sub_category = similar_meta['subCategory']
                image_filename = os.path.basename(img_path)
                st.write(f"{item_name}\\n{category} > {sub_category}\\n{image_filename}")
            else:
                st.write("Metadata not found.")

    if nb_closest_images > len(cols):
        num_rows = ((nb_closest_images - len(cols)) // len(cols) + 1) - 1
        for row in range(num_rows):
            cols = st.columns(len(cols))
            for j in range(row * len(cols), min((row + 1) * len(cols), nb_closest_images)):
                with cols[j - row * len(cols)]:
                    st.image(similar_images[j], caption="Similar", width=200)
                    similar_meta = df[df["image_path"].str.strip() == similar_images[j].strip()]
                    if not similar_meta.empty:
                        similar_meta = similar_meta.iloc[0]
                        item_name = similar_meta['productDisplayName']
                        category = similar_meta['masterCategory']
                        sub_category = similar_meta['subCategory']
                        image_filename = os.path.basename(similar_images[j])
                        st.write(f"{item_name}\\n{category} > {sub_category}\\n{image_filename}")
                    else:
                        st.write("Metadata not found.")

# --- Google Image Search Functions (Second Interface) ---
def search_google_images(query, num_results=5):
    service = build("customsearch", "v1", developerKey=API_KEY)
    res = service.cse().list(q=query, cx=CSE_ID, searchType='image', num=num_results).execute()
    try:
        return res['items']
    except KeyError:
        return []

def extract_features_google(img_pil):
    try:
        img = img_pil.resize((224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))
        resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        feature_extractor = Model(inputs=resnet_model.input, outputs=resnet_model.output)
        features = feature_extractor.predict(img_array)
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def get_similar_image_links_and_display(image_bytes):
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content(["Give 3-4 words describing this image.", pil_image])
        image_description = response.text
        st.write(f"Gemini Description (Search Query): {image_description}")

        search_results = search_google_images(image_description)

        if not search_results:
            st.write("No links found.")
            return

        st.write("Image Links andMetadata:")
        for item in search_results:
            st.write(f"Link: {item['link']}")
            st.write(f"Title: {item.get('title', 'N/A')}")
            st.write(f"Snippet: {item.get('snippet', 'N/A')}")
            st.write(f"Display Link: {item.get('displayLink', 'N/A')}")
            st.write("---")

        st.image(pil_image, caption="Uploaded Image")

        for item in search_results:
            try:
                img_data = requests.get(item['link']).content
                img_pil = Image.open(io.BytesIO(img_data))
                st.image(img_pil)
            except Exception as e:
                st.error(f"Error displaying image: {e}")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Streamlit UI with Toggle ---
def main():
    st.title("Visual Search")

    interface_toggle = st.sidebar.radio("Select Interface", ("Local Image Search", "Google Image Search"))

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()

        if interface_toggle == "Local Image Search":
            temp_img_path = "temp_image.jpg"
            with open(temp_img_path, "wb") as f:
                f.write(image_bytes)
            show_results(temp_img_path)
            os.remove(temp_img_path)
        else:
            if st.button("Search"):
                get_similar_image_links_and_display(image_bytes)
    else:
        st.write("Please upload an image.")

if __name__ == "__main__":
    main()

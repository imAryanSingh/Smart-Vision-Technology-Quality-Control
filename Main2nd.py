import streamlit as st
from PIL import Image, ImageOps  # Make sure to install Pillow
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import os
import pytesseract
import streamlit as st
from inference_sdk import InferenceHTTPClient
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

import cv2
from pyzbar.pyzbar import decode
import sqlite3
import os

# =========================
# Streamlit App Configuration
# =========================
st.set_page_config(page_title="Smart Vision Technology Quality Control", layout="wide")

# Sidebar for navigation options
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select Task", ["Project Details", "Solution Summary", "Freshness Index", "Product Recognition","Detail Extraction","Count Fruit&Vege"])

# =========================
# Project Details Page
# =========================
if app_mode == "Project Details":

    # Background Image
    background_image = rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Images\Photo\Designer.png"  # Update with your image path

    # Add background image using CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({background_image});
            background-size: cover;
            background-repeat: no-repeat;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and Introduction
    st.title("Smart Vision Technology Quality Control")
    st.write("""
             This application aims to revolutionize quality testing using advanced camera vision technology for India's largest e-commerce company.
             The goal is to design a smart quality test system that effectively assesses shipment quality and quantity.
             """)

    # =========================
    # Insert and Resize Image
    # =========================
    image_path = rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Images\Photo\Designer.png"  # Update with your image path

    # Open and resize the image
    image = Image.open(image_path)
    image = image.resize((800, 400))  # Resize to 800x400 pixels

    # Display the resized image
    st.image(image, caption='Smart Vision Technology in Action', use_column_width=True)

    # =========================
    # Project Overview
    # =========================
    st.header("Project Overview")
    st.write("""
    The project focuses on automating the quality inspection process through smart vision technology. 
    This involves identifying products, assessing their quantity, and detecting any defects or quality attributes. 
    The key features of the Smart Vision System include:
    """)

    # Key Features List with Icons for Visual Appeal
    key_features = [
        "🖼️ **Image Acquisition**: Utilizing high-resolution cameras with controlled lighting to capture clear images of products.",
        "🔍 **Image Preprocessing**: Normalizing images for consistency, applying filters to enhance features, and segmenting images for meaningful analysis.",
        "📊 **Feature Extraction**: Implementing OCR for text detection, analyzing geometric features, and applying object detection algorithms.",
        "🤖 **Classification and Decision-Making**: Using machine learning models to classify products based on extracted features and comparing them against a product database.",
        "📈 **Output and Feedback**: Providing real-time feedback on product quality, logging data for analysis, and improving processes over time.",
        "🔗 **Integration with Existing Systems**: Automating handling processes and connecting to inventory management systems."
    ]

    for feature in key_features:
        st.write(feature)

    # =========================
    # Applications in E-commerce
    # =========================
    st.header("Applications in E-commerce")
    st.write("""
    The Smart Vision System can be applied in various ways within the e-commerce industry:
    """)

    applications = [
        "1. 🏷️ **Item Recognition**: Identifying unique products, categories, and counts.",
        "2. 📦 **Packaging Inspection**: Verifying the integrity and correctness of packaging and labeling.",
        "3. 🕒 **Expiration Date Verification**: Ensuring products are fresh by recognizing printed expiration dates.",
        "4. 🍏 **Fresh Produce Inspection**: Automatically assessing the quality of fruits and vegetables by detecting defects or irregular shapes.",
        "5. 📊 **Bin Monitoring**: Monitoring stock levels and ensuring proper placement of products in inventory."
    ]

    for app in applications:
        st.write(app)

    # =========================
    # Challenges in Implementation
    # =========================
    st.header("Challenges in Implementing Smart Vision")
    st.write("""
    While implementing smart vision technology, several challenges may arise:
    """)

    challenges = [
        "🌤️ **Environmental Conditions**: Variability in lighting and background can affect image quality.",
        "🔄 **Complexity of Products**: Variability in size, shape, and color complicates analysis.",
        "💰 **Cost and Integration**: Balancing technology costs with benefits while integrating with existing systems."
    ]

    for challenge in challenges:
        st.write(challenge)

    # =========================
    # Expected Outcomes from Participants
    # =========================
    st.header("Expected Outcomes from Participants")
    st.write("""
    Participants are encouraged to develop innovative solutions that address these challenges while leveraging smart vision technology to enhance quality testing processes. 
    Consider the following aspects when designing your solution:
    """)

    outcomes = [
        "1. ✅ **Accuracy**: Ensure the system can accurately detect and classify quality attributes.",
        "2. ⚡ **Efficiency**: Minimize processing time to keep up with high-volume operations.",
        "3. 💡 **Cost-Effectiveness**: Propose solutions that are affordable and scalable.",
        "4. 👥 **User Experience**: Design systems that are easy for staff to operate and understand."
    ]

    for outcome in outcomes:
        st.write(outcome)

    # =========================
    # Conclusion
    # =========================
    st.header("Conclusion")
    st.write("""
    The Smart Vision Technology Quality Control project presents an exciting opportunity for participants to showcase their skills 
    and develop real-world solutions that can significantly impact quality testing processes in the e-commerce industry.
    """)

elif app_mode == "Solution Summary":
    background_image = rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Images\Photo\_a3090cf4-9625-4ebe-bf55-a4a8b0cf0f30 (1).jpg"  # Update with your image path

    # Add background image using CSS
    st.markdown(
        f"""
            <style>
            .stApp {{
                background-image: url({background_image});
                background-size: cover;
                background-repeat: no-repeat;
                color: white;
            }}
            </style>
            """,
        unsafe_allow_html=True
    )

    # Title and Introduction
    st.title("Proposed Solution Overview Of Smart Vision Technology Quality Control")
    st.write("""
             This application aims to transform quality testing by leveraging cutting-edge camera vision technology for India's largest e-commerce company.
             The objective is to create an intelligent quality assessment system that accurately evaluates both the quality and quantity of shipments.
             """)

    # =========================
    # Insert and Resize Image for Solution Summary
    # =========================
    solution_image_path = rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Images\Photo\_a3090cf4-9625-4ebe-bf55-a4a8b0cf0f30 (1).jpg"  # Update with your image path

    # Open and resize the image
    solution_image = Image.open(solution_image_path)
    solution_image = solution_image.resize((800, 400))  # Resize to 800x400 pixels

    # Display the resized image
    st.image(solution_image, caption='Proposed Solution Overview', use_column_width=True)

    # =========================
    # Solution Summary Section
    # =========================
    st.header("Solution Summary")
    st.write("""
    The proposed solution employs state-of-the-art imaging technologies and sophisticated algorithms to capture and analyze visual data for quality control in e-commerce. 
    The primary components of this solution are detailed below:
    """)

    # Key Components of the Solution with Icons for Visual Appeal
    key_components = [
        "📷 **Image Acquisition**: Utilizes high-resolution cameras with optimal lighting to ensure clear images of products during inspection.",

        "🔍 **Image Preprocessing**: Normalizes images for brightness, contrast, and color balance, applying filters to enhance features and segment relevant areas.",

        "🧠 **Feature Extraction**: Implements Optical Character Recognition (OCR) for text detection, analyzes geometric features, and employs object detection algorithms to identify defects or quality indicators.",

        "🤖 **Classification and Decision-Making**: Utilizes machine learning models to classify products based on extracted features, comparing against a database of known items.",

        "📈 **Output and Feedback**: Provides real-time feedback on product quality, logging data for further analysis and continuous process improvement.",

        "🔗 **Integration with Existing Systems**: Seamlessly integrates with conveyor systems and inventory management for automated handling based on quality evaluations."
    ]

    for component in key_components:
        st.write(component)

    # =========================
    # Applications in E-commerce Section
    # =========================
    st.header("Applications in E-commerce")
    st.write("""
    The Smart Vision System can be utilized in various capacities within the e-commerce sector:
    """)

    applications = [
        "1. 🏷️ **Item Recognition**: Accurately identifying unique products, categories, and quantities.",
        "2. 📦 **Packaging Inspection**: Ensuring the integrity and correctness of packaging and labeling.",
        "3. 🕒 **Expiration Date Verification**: Confirming product freshness by recognizing printed expiration dates.",
        "4. 🍏 **Fresh Produce Inspection**: Automatically evaluating the quality of fruits and vegetables by detecting defects or irregular shapes.",
        "5. 📊 **Bin Monitoring**: Keeping track of stock levels and ensuring proper product placement in inventory."
    ]

    for app in applications:
        st.write(app)

    # =========================
    # Challenges in Implementation Section
    # =========================
    st.header("Challenges in Implementing Smart Vision")
    st.write("""
    Several challenges may arise during the implementation of smart vision technology:
    """)

    challenges = [
        "🌤️ **Environmental Conditions**: Variability in lighting and background can significantly impact image quality.",
        "🔄 **Complexity of Products**: Differences in size, shape, and color can complicate analysis.",
        "💰 **Cost and Integration**: Balancing technology costs with benefits while ensuring smooth integration with existing systems."
    ]

    for challenge in challenges:
        st.write(challenge)

    # =========================
    # Expected Outcomes from Participants Section
    # =========================
    st.header("Expected Outcomes from Participants")
    st.write("""
    Participants are encouraged to devise innovative solutions that tackle these challenges while harnessing smart vision technology to enhance quality testing processes. 
    Consider these aspects when designing your solution:
    """)

    outcomes = [
        "1. ✅ **Accuracy**: Ensure the system can reliably detect and classify quality attributes.",
        "2. ⚡ **Efficiency**: Reduce processing time to accommodate high-volume operations.",
        "3. 💡 **Cost-Effectiveness**: Propose solutions that are both affordable and scalable.",
        "4. 👥 **User Experience**: Create systems that are intuitive for staff to operate."
    ]

    for outcome in outcomes:
        st.write(outcome)

    # =========================
    # Conclusion Section
    # =========================
    st.header("Conclusion")
    st.write("""
    The Smart Vision Technology Quality Control project offers a unique opportunity for participants to demonstrate their skills 
    and develop practical solutions that can greatly enhance quality testing processes within the e-commerce industry.
    """)
elif app_mode == "Freshness Index":

    background_image = rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Images\Photo\home_img.jpg"  # Update with your image path
    # Sidebar for navigation options

    # Add background image using CSS
    st.markdown(
        f"""
            <style>
            .stApp {{
                background-image: url({background_image});
                background-size: cover;
                background-repeat: no-repeat;
                color: white;
            }}
            </style>
            """,
        unsafe_allow_html=True
    )


    # =========================
    # Custom DepthwiseConv2D Layer
    # =========================
    class CustomDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            if 'groups' in kwargs:
                kwargs.pop('groups')
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, config):
            if 'groups' in config:
                config.pop('groups')
            return cls(**config)


    # =========================
    # Streamlit App Configuration
    # =========================

    st.title("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Images\Photo\home_img.jpg"  # Update with your image path
    image = Image.open(image_path)
    image = image.resize((800, 400))  # Resize to 800x400 pixels

    # Display the resized image
    st.image(image, caption='Fruit and Vegetable Recognision', use_column_width=True)
    st.write("""
    The **Freshness Index Detection** is a critical component of the Smart Vision Technology Quality Control project. 
    This system aims to assess the freshness of perishable products, such as fruits and vegetables, by analyzing various visual cues and patterns.
                 """)
    st.write("""
    It can check **Freshness Status ** of Bellpepper, Carrot, Tomato, Cucumber, Potato, Mango, Apple, Strawberry, Orange, Banana
    \n plus (+)
    \n It can **Detect** several fruit's and vegetable's like Garlic, Grapes, Jalepeno, Onion, Pear, Pomegranate, Turnip, Watermelon, Beetroot, Cabbage, Corn, Ginger, Kiwi, Lemon, Peas, Pineapple, Cauliflower, Chili Pepper, Eggplant
          """)


    # =========================
    # Load Model Function with Caching
    # =========================
    @st.cache_resource
    def load_trained_model(model_path):
        custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return model


    # =========================
    # Load Labels Function with Caching
    # =========================
    @st.cache_data
    def load_labels(labels_path):
        with open(labels_path, "r") as f:
            class_names = f.read().splitlines()
        return class_names


    # =========================
    # Paths to Model and Labels
    # =========================
    # Update these paths as per your directory structure
    MODEL_PATH = rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\converted_keras\keras_model.h5"
    LABELS_PATH = rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\converted_keras\labels.txt"

    # Check if model and labels exist
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
        st.stop()

    if not os.path.exists(LABELS_PATH):
        st.error(f"Labels file not found at {LABELS_PATH}. Please check the path.")
        st.stop()

    # =========================
    # Load Model and Labels
    # =========================
    with st.spinner('Loading model...'):
        model = load_trained_model(MODEL_PATH)
        class_names = load_labels(LABELS_PATH)

    st.success('Model loaded successfully!')

    # Optionally display model summary
    if st.checkbox("Show Model Summary"):
        with st.expander("Model Summary"):
            # Capture the model summary
            import io
            import sys

            buffer = io.StringIO()
            sys.stdout = buffer
            model.summary()
            sys.stdout = sys.__stdout__
            summary = buffer.getvalue()
            st.text(summary)

    # =========================
    # Image Upload
    # =========================
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Open and display the image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            # =========================
            # Image Preprocessing
            # =========================
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.expand_dims(normalized_image_array, axis=0)

            # =========================
            # Make Prediction
            # =========================
            with st.spinner('Making prediction...'):
                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index].strip()
                confidence_score = prediction[0][index]

            # =========================
            # Display Prediction
            # =========================
            st.success('Prediction completed!')
            st.write(f"**Class:** {class_name[2:]}")
            st.write(f"**Confidence Score:** {confidence_score:.4f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload an image to get started.")

elif app_mode == "Product Recognition":
    background_image = rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Images\Photo\Designer.jpeg"  # Update with your image path

    # Add background image using CSS
    st.markdown(
        f"""
            <style>
            .stApp {{
                background-image: url({background_image});
                background-size: cover;
                background-repeat: no-repeat;
                color: white;
            }}
            </style>
            """,
        unsafe_allow_html=True
    )

    # Replace with your Roboflow API key
    api_key = "2KTKv5dLpka5kKZbud3g"

    # Initialize the Roboflow Inference Client
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

    st.header("PRODUCT RECOGNITION SYSTEM")
    st.write("""
    The **Product Recognition** feature is a crucial aspect of our Smart Vision Technology Quality Control system. 
    This functionality utilizes advanced imaging systems to identify products accurately based on their unique features.
    \n
    **Products it can recognise** are Ariel, Coca Cola, Colgate, Fanta, kurkure, Lays Masala, Lays mexican, Lifebuoy Soap, Sunsilk Shampoo, Vaseline lotion, Chocos, Colgate, Complan, Glucon D, Hamam Soap, Horlicks, AVT Tea, Boost, Bourvita, Brookbond Redlabel, Brookbond tajmahal""")

    image = Image.open(rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Images\Photo\Designer.jpg")
    image = image.resize((800, 400))
    st.image(image, caption='Product recognision System in Action', use_column_width=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # Convert bytes data to base64 string
        base64_image = base64.b64encode(bytes_data).decode('utf-8')

        model_ids = ["shop_stock_dataset/1", "jg-intern/4"]
        results = {}

        for model_id in model_ids:
            results[model_id] = CLIENT.infer(base64_image, model_id=model_id)

        label_counts = {}

        for model_id, result in results.items():
            if 'predictions' in result and result['predictions']:
                for prediction in result['predictions']:
                    label = prediction['class']
                    confidence = prediction['confidence']

                    # Count occurrences and store confidence scores
                    if label in label_counts:
                        label_counts[label]['count'] += 1
                        label_counts[label]['confidences'].append(confidence)
                    else:
                        label_counts[label] = {'count': 1, 'confidences': [confidence]}

        # Display image with bounding boxes
        if label_counts:
            image_data_decoded = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data_decoded))

            fig, ax = plt.subplots(1)
            ax.imshow(image)

            for model_id, result in results.items():
                if 'predictions' in result and result['predictions']:
                    predictions = result['predictions']
                    for prediction in predictions:
                        x = prediction['x']
                        y = prediction['y']
                        width = prediction['width']
                        height = prediction['height']
                        class_name = prediction['class']
                        confidence = prediction['confidence']

                        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        plt.text(x, y, f"{class_name} ({confidence:.2f})", color='red', fontsize=8)

            st.pyplot(fig)

            # Display label occurrences and confidence scores
            st.write("\nLabel Occurrences with Confidence Scores:")
            for label, data in label_counts.items():
                avg_confidence = sum(data['confidences']) / len(data['confidences'])
                st.write(f"Label: {label}, Count: {data['count']}, Average Confidence: {avg_confidence * 100:.2f}%")
        else:
            st.write("No predictions found in any of the models.")

elif app_mode == "Count Fruit&Vege":

    background_image = rf"C:\Users\aryan\Downloads\Designer (1).jpg"  # Update with your image path

    # Add background image using CSS
    st.markdown(
        f"""
                <style>
                .stApp {{
                    background-image: url({background_image});
                    background-size: cover;
                    background-repeat: no-repeat;
                    color: white;
                }}
                </style>
                """,
        unsafe_allow_html=True
    )

    # Replace with your Roboflow API key
    api_key = "2KTKv5dLpka5kKZbud3g"

    # Initialize the Roboflow Inference Client
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

    st.header("Fruit and Vegetable Counting Application")
    st.write("""
    Welcome to the **Fruit and Vegetable Counting Application**! This tool leverages advanced image recognition technology to accurately count and identify various fruits and vegetables in your images. 
    \n 
    It can **Count Fruits and Vegetable's** are apple, beans, beetroot, bell_pepper, cabbage, carrot, cucumber, egg, eggplant, garlic, grape, lemon, mango, napa cabbage, onion, orange, peach, pepper, potato, radish, sapota, tomato, turnip, zuchhini
    """)

    image = Image.open(rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Designer (1) Fruit.jpg")
    image = image.resize((800, 400))
    st.image(image, caption='Counting Images', use_column_width=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # Convert bytes data to base64 string
        base64_image = base64.b64encode(bytes_data).decode('utf-8')

        model_ids = ["tdl_final-2rhq7/2"]
        results = {}

        for model_id in model_ids:
            results[model_id] = CLIENT.infer(base64_image, model_id=model_id)

        label_counts = {}

        for model_id, result in results.items():
            if 'predictions' in result and result['predictions']:
                for prediction in result['predictions']:
                    label = prediction['class']
                    confidence = prediction['confidence']

                    # Count occurrences and store confidence scores
                    if label in label_counts:
                        label_counts[label]['count'] += 1
                        label_counts[label]['confidences'].append(confidence)
                    else:
                        label_counts[label] = {'count': 1, 'confidences': [confidence]}

        # Display image with bounding boxes
        if label_counts:
            image_data_decoded = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data_decoded))

            fig, ax = plt.subplots(1)
            ax.imshow(image)

            for model_id, result in results.items():
                if 'predictions' in result and result['predictions']:
                    predictions = result['predictions']
                    for prediction in predictions:
                        x = prediction['x']
                        y = prediction['y']
                        width = prediction['width']
                        height = prediction['height']
                        class_name = prediction['class']
                        confidence = prediction['confidence']

                        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        plt.text(x, y, f"{class_name} ({confidence:.2f})", color='red', fontsize=8)

            st.pyplot(fig)

            # Display label occurrences and confidence scores
            st.write("\nLabel Occurrences with Confidence Scores:")
            for label, data in label_counts.items():
                avg_confidence = sum(data['confidences']) / len(data['confidences'])
                st.write(f"Label: {label}, Count: {data['count']}, Average Confidence: {avg_confidence * 100:.2f}%")
        else:
            st.write("No predictions found in any of the models.")

elif app_mode == "Detail Extraction":

    # Set the path for Tesseract-OCR if it's not in PATH
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path as needed


    # =========================
    # Custom DepthwiseConv2D Layer
    # =========================
    class CustomDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, config):
            config.pop('groups', None)
            return cls(**config)


    # =========================
    # Load Model Function with Caching
    # =========================
    @st.cache_resource
    def load_trained_model(model_path):
        custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        return model


    # =========================
    # Load Labels Function with Caching
    # =========================
    @st.cache_data
    def load_labels(labels_path):
        with open(labels_path, "r") as f:
            return f.read().splitlines()


    # =========================
    # Database Functions
    # =========================

    def create_database():
        conn = sqlite3.connect(
            r'C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\products.db')  # Update with your database path
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                MRP REAL,
                BrandName TEXT,
                SizeDetail TEXT,
                ManufacturingDate TEXT,
                ExpiryDate TEXT,
                ProductName TEXT UNIQUE  -- Ensure this matches your query for product name.
            )
        ''')

        conn.commit()
        conn.close()


    def insert_product_details(mrp, brand_name, size_detail, manufacturing_date, expiry_date, product_name):
        conn = sqlite3.connect(
            r'C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\products.db')  # Update with your database path
        cursor = conn.cursor()

        # Insert product details; use a parameterized query to prevent SQL injection.
        cursor.execute('''
            INSERT OR IGNORE INTO Products (MRP, BrandName, SizeDetail, ManufacturingDate, ExpiryDate, ProductName)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (mrp, brand_name, size_detail, manufacturing_date, expiry_date, product_name))

        conn.commit()
        conn.close()


    def query_product_details(product_name):
        conn = sqlite3.connect(
            r'C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\products.db')  # Update with your database path
        cursor = conn.cursor()

        # Query to fetch product details based on the product name.
        cursor.execute(
            "SELECT MRP, BrandName, SizeDetail, ManufacturingDate, ExpiryDate FROM Products WHERE ProductName = ?",
            (product_name,))
        result = cursor.fetchone()

        conn.close()

        return result


    # =========================
    # Main Application Code
    # =========================

    background_image = rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Images\Photo\_b5b86181-4320-46c0-b2c4-0f4f8ab27ad0.jpg"
    st.markdown(
        f"""
            <style>
            .stApp {{
                background-image: url({background_image});
                background-size: cover;
                background-repeat: no-repeat;
                color: white;
            }}
            </style>
            """,
        unsafe_allow_html=True
    )

    api_key = "2KTKv5dLpka5kKZbud3g"  # Replace with your Roboflow API key

    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

    st.header("Detail Extraction System")
    st.write("""
    The **Detail Extraction** feature is a vital component of our Smart Vision Technology Quality Control system. 
    This functionality harnesses the power of Optical Character Recognition (OCR) and advanced image processing techniques to extract critical information from product images.""")
    st.write("\n**Products it can Extract Detail's(Saved In database) of** are Ariel, Coca Cola, Colgate, Fanta, kurkure, Lays Masala, Lays mexican, Lifebuoy Soap, Sunsilk Shampoo, Vaseline lotion, Chocos, Colgate, Complan, Glucon D, Hamam Soap, Horlicks, AVT Tea, Boost, Bourvita, Brookbond Redlabel, Brookbond tajmahal""")
    image = Image.open(rf"C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\Images\Photo\_b5b86181-4320-46c0-b2c4-0f4f8ab27ad0.jpg")
    image = image.resize((800, 400))
    st.image(image, caption='Detail Extraction in Action', use_column_width=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        base64_image = base64.b64encode(bytes_data).decode('utf-8')

        model_ids = ["shop_stock_dataset/1", "jg-intern/4"]
        results = {}

        for model_id in model_ids:
            results[model_id] = CLIENT.infer(base64_image, model_id=model_id)

        label_counts = {}

        # Find the product with the highest confidence score.
        highest_confidence_product = None
        highest_confidence_score = 0

        for model_id, result in results.items():
            if 'predictions' in result and result['predictions']:
                for prediction in result['predictions']:
                    label = prediction['class']
                    confidence = prediction['confidence']

                    # Update highest confidence product.
                    if confidence > highest_confidence_score:
                        highest_confidence_product = label
                        highest_confidence_score = confidence

                    # Count occurrences and store confidence scores.
                    if label in label_counts:
                        label_counts[label]['count'] += 1
                        label_counts[label]['confidences'].append(confidence)
                    else:
                        label_counts[label] = {'count': 1, 'confidences': [confidence]}

        # Display image with bounding boxes.
        if label_counts:
            image_data_decoded = base64.b64decode(base64_image)
            image_displayed = Image.open(io.BytesIO(image_data_decoded))

            fig, ax = plt.subplots(1)
            ax.imshow(image_displayed)

            for model_id, result in results.items():
                if 'predictions' in result and result['predictions']:
                    predictions = result['predictions']
                    for prediction in predictions:
                        x = prediction['x']
                        y = prediction['y']
                        width = prediction['width']
                        height = prediction['height']
                        class_name = prediction['class']
                        confidence = prediction['confidence']

                        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        plt.text(x, y, f"{class_name} ({confidence:.2f})", color='red', fontsize=8)

            st.pyplot(fig)

            # Display details of the highest confidence product.
            if highest_confidence_product:
                st.write(
                    f"**Highest Confidence Product:** {highest_confidence_product} ({highest_confidence_score:.2f})")

                # Query the database for additional details.
                product_details = query_product_details(highest_confidence_product)
                if product_details:
                    st.write(f"**MRP:** {product_details[0]}")
                    st.write(f"**Brand Name:** {product_details[1]}")
                    st.write(f"**Size Detail:** {product_details[2]}")
                    st.write(f"**Manufacturing Date:** {product_details[3]}")
                    st.write(f"**Expiry Date:** {product_details[4]}")
                else:
                    st.write("No additional details found for this product.")

    else:
        st.info("Please upload an image to get started.")

    # Example usage of creating the database and inserting sample data (uncomment to use):
    if __name__ == "__main__":
        create_database()  # Create the database and table
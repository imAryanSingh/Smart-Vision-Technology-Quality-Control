# Flipkart Grid 6.0 - Smart Vision Technology Quality Control Project

## Overview

This repository contains the code developed for the **Smart Vision Technology Quality Control** project, which aims to revolutionize quality testing in e-commerce using advanced camera vision technology. The project focuses on automating the quality inspection process by identifying products, assessing their quantity, and detecting defects or quality attributes.

## Project Description

The Smart Vision Technology Quality Control system is designed to enhance the quality assurance processes for one of India’s largest e-commerce companies. By utilizing high-resolution imaging and sophisticated algorithms, the system can accurately evaluate both the quality and quantity of shipments, ensuring that customers receive only the best products.

## Features

- **Image Acquisition**: High-resolution cameras with controlled lighting capture clear images of products.
- **Image Preprocessing**: Normalizes images for consistency and applies filters to enhance features.
- **Feature Extraction**: Implements Optical Character Recognition (OCR) for text detection and uses object detection algorithms to identify defects.
- **Classification and Decision-Making**: Utilizes machine learning models to classify products based on extracted features.
- **Output and Feedback**: Provides real-time feedback on product quality and logs data for continuous improvement.
- **Integration with Existing Systems**: Seamlessly connects with inventory management systems for automated handling based on quality evaluations.
- **Roboflow Integration**: Utilizes the Roboflow API for model deployment, enabling efficient image processing and product recognition through pre-trained models.

## Requirements

- Python 3.7+
- Streamlit
- TensorFlow
- OpenCV
- Pillow
- NumPy
- PyZbar
- SQLite3
- Roboflow API (for model inference)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/imAryanSingh/Smart-Vision-Technology-Quality-Control.git
   cd Smart-Vision-Technology-Quality-Control
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure all necessary hardware components (like cameras) are connected and configured correctly.

4. Set up your Roboflow account and obtain an API key to access pre-trained models.

## Usage

1. **Project Details**:
   The application provides an overview of its objectives and key features, showcasing how it aims to automate quality inspections.

2. **Solution Summary**:
   This section details how advanced imaging technologies are utilized for quality control in e-commerce.

3. **Freshness Index**:
   Users can upload images of fruits and vegetables to assess their freshness based on visual analysis.

4. **Product Recognition**:
   The system identifies products from uploaded images using machine learning models trained on various product categories via Roboflow's API.

5. **Detail Extraction**:
   This feature extracts relevant information from product images using OCR techniques.

6. **Count Fruits & Vegetables**:
   Users can count items within an uploaded image using object detection algorithms.

To run the project, execute the main script:
```bash
streamlit run main2nd.py
```

## State Machine

The application operates in various states depending on user interactions:
- **INITIALIZING**: Sets up the application and loads necessary models.
- **IMAGE_UPLOAD**: Allows users to upload images for analysis.
- **PROCESSING**: Analyzes uploaded images for freshness, recognition, or detail extraction.
- **RESULT_DISPLAY**: Shows results from the analysis back to the user.

## License and Permissions

This code is made publicly available for reference purposes in my resume. You are free to refer to it, but you may not use or modify the code without explicit permission.

## Contact

For any inquiries or permissions, please contact:

- **Name**: Aryan Singh
- **Email**: aryansingh4653@gmail.com
- **Phone Number**: +91-8955424401
- **LinkedIn**: [Aryan Singh](https://www.linkedin.com/in/im-aryan-singh/)

## Team Members

- **[Pranjal Galundia](https://www.linkedin.com/in/pranjal-galundia-806823319/)** (SVKM's NMIMS Mukesh Patel School of Technology Management & Engineering, Shirpur Campus)
- **[Aryan Singh](https://www.linkedin.com/in/im-aryan-singh/)** (Institute of Engineering and Technology, Udaipur)

We are students from [Vidhya Institute Of Information Technology, Udaipur](+91-9214465362).

## Acknowledgments

We would like to thank Flipkart for providing this opportunity to showcase our skills through this project, as well as Roboflow for their powerful tools that facilitate model deployment and inference.

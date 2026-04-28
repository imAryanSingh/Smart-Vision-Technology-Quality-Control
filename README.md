# Smart Vision Technology Quality Control System
 

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-API-6706CE?style=for-the-badge)
![Status](https://img.shields.io/badge/Flipkart_GRID_6.0-Top_0.3%25-1D9E75?style=for-the-badge)

**An end-to-end automated quality control pipeline for e-commerce — product recognition, freshness detection, OCR extraction, and fruit/vegetable counting from a single image upload.**

*Ranked Top 0.3% (300 of 1,00,000+ participants) in Flipkart GRID 6.0 · Reduced manual inspection effort by 40%*
 
[Overview](#overview) · [demo](#demo) · [Pipeline](#pipeline) · [Results](#results) · [Setup](#setup) · [File Structure](#file-structure)
 
## Overview
 
Manual quality inspection in e-commerce warehouses is slow, inconsistent, and unscalable. This system automates the entire QC workflow using computer vision — a warehouse worker uploads an image, and the system instantly returns product identity, freshness score, text details, and item count.
 
Built for **Flipkart GRID 6.0** (Software Development + Robotics tracks), this project achieved **Top 0.3% nationally** out of 1,00,000+ participants.
 
| Capability | Method | Accuracy |
|------------|--------|----------|
| Product recognition | Roboflow trained model | 20+ product types |
| Freshness assessment | Custom CNN (Keras) | Fresh / Rotten classification |
| Text extraction (OCR) | Tesseract + OpenCV | Label & barcode details |
| Item counting | Object detection | 30+ fruit/vegetable categories |
| Data logging | SQLite3 | Persistent inspection records |
 
---
 
## Demo

The following video demonstrates the end-to-end capabilities of the system, including real-time freshness detection and OCR extraction:

**🔗 Video Link:** [Flipkart GRID 6.0 - Smart Vision Technology Demo](https://www.linkedin.com/posts/im-aryan-singh_shapingindiantechscape-flipkartgrid-jointhegrid-ugcPost-7262143225430253569-_Ecb)

**What's shown in the video:**
* **Real-time Inference:** Fast processing of uploaded warehouse images.
* **Accuracy:** High-confidence freshness scoring for perishables.
* **Workflow:** Seamless integration between the Streamlit UI and the backend ML models.
 
**App sections:**
| Page | What it does |
|------|-------------|
| Project Details | Overview of objectives and features |
| Freshness Index | Upload fruit/vegetable → get Fresh/Rotten + confidence |
| Product Recognition | Upload product image → identify from 20+ categories |
| Detail Extraction | OCR on product labels → extract name, expiry, price |
| Count Items | Upload image → count fruits/vegetables using detection |
 
---
 
## Pipeline
 
```
┌──────────────────────────────────────────────────────────────────┐
│              SMART VISION QC PIPELINE                            │
├──────────────┬───────────────┬───────────────┬──────────────────┤
│   IMAGE      │  PREPROCESS   │   INFERENCE   │    OUTPUT        │
│   UPLOAD     │               │               │                  │
│  Streamlit   │  OpenCV       │  Roboflow API │  Result display  │
│  UI          │  Resize       │  TensorFlow   │  SQLite log      │
│  JPG/PNG     │  Normalize    │  Tesseract    │  CSV export      │
└──────────────┴───────────────┴───────────────┴──────────────────┘
```
 
### Module 1 — Freshness Detection
 
```python
# Custom Keras CNN trained on fresh vs rotten fruit dataset
# Input: image → Output: [Fresh, Rotten] + confidence score
model = load_model('converted_keras/keras_model.h5')
labels = open('converted_keras/labels.txt').read().splitlines()
 
img = Image.open(uploaded_file).resize((224, 224))
img_array = np.array(img) / 255.0
prediction = model.predict(img_array[np.newaxis, ...])
# Result: "Fresh Mango (confidence: 94.2%)" or "Rotten Apple (87.1%)"
```
 
### Module 2 — Product Recognition (Roboflow)
 
```python
# Roboflow model trained on 5,000+ labelled product images
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
model = rf.workspace().project("product-recognition").version(1).model
result = model.predict(image_path, confidence=40).json()
# Returns: product name, bounding box, confidence for each detected item
```
 
### Module 3 — Detail Extraction (OCR)
 
```python
import pytesseract, cv2
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
text = pytesseract.image_to_string(thresh)
# Extracts: product name, MFG date, expiry date, MRP, barcode
```
 
### Module 4 — SQLite Logging
 
```python
# Every inspection is logged to products.db
import sqlite3
conn = sqlite3.connect('products.db')
cursor.execute('''INSERT INTO inspections
    (timestamp, product, freshness, confidence, count)
    VALUES (?, ?, ?, ?, ?)''', (ts, product, fresh, conf, count))
conn.commit()
```
 
---
 
## Results
 
| Metric | Value |
|--------|-------|
| Products classifiable | 20+ categories |
| Fruits/vegetables assessed | 30+ types |
| Training images | 5,000+ labelled |
| Manual inspection effort reduced | 40% |
| Flipkart GRID 6.0 rank | Top 0.3% (300 / 1,00,000+) |
| Competition tracks | Software Development + Robotics |
 
### Sample Detection Output
 
```
Image: mango_batch_12.jpg
─────────────────────────────────────────
Product identified:    Mango
Freshness:             Fresh (94.2% confidence)
Count:                 7 items detected
Label text (OCR):      "Alphonso Mango · MRP ₹120 · Best before 3 days"
Inspection logged:     products.db → ID #1847
─────────────────────────────────────────
Processing time:       1.8 seconds
```
 
---
 
## File Structure
 
```
Smart-Vision-Technology-Quality-Control/
│
├── Main2nd.py                      ← Main Streamlit app (entry point)
├── SQLLITE_CODE.py                 ← Database schema and logging helpers
├── products.db                     ← SQLite inspection log database
│
├── converted_keras/                ← Freshness detection model
│   ├── keras_model.h5              ← Trained Keras CNN weights
│   └── labels.txt                  ← Class labels (Fresh/Rotten + categories)
│
├── converted_keras_Extraction/     ← OCR extraction model assets
│
├── FreshCucumber/                  ← Sample images: fresh cucumber
├── RottenMango/                    ← Sample images: rotten mango
├── Download_image/                 ← Downloaded test images
├── Upload_Images/                  ← User upload buffer
├── Photo/                          ← Additional sample images
├── images/                         ← UI assets
│
├── .devcontainer/                  ← VS Code dev container config
├── requirements.txt
└── README.md
```
 
> **Recommended cleanup:**
> Rename image files in root — `Designer (1) Fruit.jpg`, `MixCollage-18-Oct-2024-09-03-PM-3920.jpg` etc. should go into the `images/` folder with clean names like `images/sample_fruit_collage.jpg`.
 
---
 
## Setup
 
### 1. Clone and install
 
```bash
git clone https://github.com/imAryanSingh/Smart-Vision-Technology-Quality-Control.git
cd Smart-Vision-Technology-Quality-Control
pip install -r requirements.txt
```
 
### 2. Set your Roboflow API key
 
```python
# In Main2nd.py, replace:
rf = Roboflow(api_key="YOUR_API_KEY")
# Get your free key at: https://roboflow.com
```
 
### 3. Run
 
```bash
streamlit run Main2nd.py
```
Opens at `http://localhost:8501`
 
### requirements.txt
 
```
streamlit>=1.20.0
tensorflow>=2.10.0
opencv-python>=4.5.0
Pillow>=9.0.0
numpy>=1.21.0
pytesseract>=0.3.10
pyzbar>=0.1.9
roboflow>=1.0.0
```
 
---
 
## Competition Context — Flipkart GRID 6.0
 
This project was submitted to **Flipkart GRID 6.0**, India's largest engineering competition for students.
 
| Track | Result |
|-------|--------|
| Software Development | Participated — Top 0.3% |
| Information Security | Participated |
| Robotics | Participated — Top 0.3% |
 
> Out of 1,00,000+ participants nationwide, this project placed in the **top 300**. Certificate of Merit awarded by Flipkart.
 
---
 
## Known Issues & Fixes
 
**Roboflow API timeout**
```python
# Increase timeout in predict() call:
result = model.predict(image_path, confidence=40, overlap=30).json()
# If still failing, check API key and project name
```
 
**Tesseract not found**
```bash
sudo apt-get install tesseract-ocr   # Linux
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```
 
**SQLite database locked**
```python
# Close any open connections before writing
conn = sqlite3.connect('products.db', timeout=10)
```
 
---
 
## About the Author
 
**Aryan Singh** — AI/ML Engineer
 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-im--aryan--singh-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/im-aryan-singh)
[![GitHub](https://img.shields.io/badge/GitHub-imAryanSingh-181717?style=flat&logo=github)](https://github.com/imAryanSingh)
[![Portfolio](https://img.shields.io/badge/Portfolio-imAryanSingh.github.io-534AB7?style=flat)](https://imAryanSingh.github.io)
 
*B.Tech CSE · Mohanlal Sukhadia University · GATE 2026 (88.31 percentile)*
 
---
 
## License
 
This code is publicly available for reference. You may not use or modify it without explicit written permission from the author.
 
---
 
## Also see
 
- [Wake-Word Detection — ISRO TRISHNA Satellite](https://github.com/imAryanSingh/Wakeup-Word-Detection-Model-for-voice-commanding-system)
- [Wildfire Prediction from Satellite Imagery](https://github.com/imAryanSingh/Wildfire-Prediction-Using-Satellite-Image-GSoC)
- [E-Commerce Sales Dashboard](https://github.com/imAryanSingh/E-COMMERCE-SALES-DASHBOARD)
 

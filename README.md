# 🏆 Face Recognition System  

TThis project implements a **Face Recognition System** that can detect and classify faces in images. It utilizes machine learning models for **real vs. fake face** detection. The system is designed to identify individuals and determine if a face is real or manipulated using deep learning techniques.

---
## Features

- **Face Detection**: Detects faces in images and assigns labels to them.
- **Real vs. Fake Classification**: Classifies faces as either **real** or **fake** to prevent takeing attendance using images from phones .
- **Custom Training**: Trains a YOLO model on custom datasets for face recognition.


# Prerequisites

Before running the project, ensure that the following are installed on your environment:

- Python 3.x
- pip (Python package installer)
- Required libraries:
  - `ultralytics` (YOLO)
  - `torch`
  - `opencv-python`
  - `matplotlib`
  - `pillow`

You can install the required libraries using the following command:
```bash
pip install requirments.txt
```

## 📂 Project Structure  
```
📁 face-recognition-project
│── 📄 train.py # Train and save face encodings
│── 📄 test.py # Test face recognition on images
│── 📄 add_person.py # Add a new person to the trained model
│── 📂 images/ # Directory for storing images
│── 📄 setup.sh # some bash setups
│── 📄 requirements.txt # Dependencies for running the project
│── 📄 README.md # Project documentation

```

---

## 🚀 Features  
✅ Train a face recognition model from images  
✅ Recognize faces in test images  
✅ Dynamically add new people to the model  
✅ Uses `pickle` to store face encodings  
✅ Supports real-time image processing  

---

## 🛠 Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/AllamElshekh/face-recognition-project.git  
cd face-recognition-project  
pip install -r requirements.txt  
```

## 📌 Usage
1️⃣ Train the Model
Train the system with known face images:

```
python train_faces.py --train_path images/train
```
2️⃣ Recognize Faces 

```
python test_using_fake_or_real.py --test_path images/test

```

3️⃣ Add a New Person
```
python add_person.py --image_path images/new_person.jpg --name "John Doe"
```
## 📌 Recogntion Example 
![Face Recognition Example](/output_rgb.jpg)

## 📌 Dependencies
-opencv-python

-face-recognition

-numpy

-pickle


## ⚠️ Known Issues & TPU Dependency
# ❓ Why Does This Project Work Only on v2-8 TPU in Google Colab?
-This project currently runs only on v2-8 TPU in Colab due to:

-TensorFlow TPU Optimizations

-The face recognition model may include TPU-optimized operations.

-TPUs require a specific TensorFlow version (>=2.8).

-Dlib + CUDA Conflicts with TPU

-The face_recognition library relies on dlib, which is optimized for CUDA GPUs.

-TPUs do not support CUDA, causing compatibility issues.

-Model Checkpoints Using TPU Metadata

-If the .pkl model file was trained on a TPU, it may store TPU-specific metadata.

-This can make the model incompatible with CPU/GPU environments.
## 👨‍💻 Author

**Developed by Allam Abdelmawgoud Ahmed 🚀 and hadeer bader**





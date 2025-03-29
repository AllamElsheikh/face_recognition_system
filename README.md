# 🏆 Face Recognition System  

A simple yet effective **face recognition system** using OpenCV and `face_recognition`. This project allows you to **train** a model with known faces, **recognize** faces in test images, and **dynamically add new people** to the model.

---

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
python recognize_faces.py --test_path images/test

```

3️⃣ Add a New Person
```
python add_person.py --image_path images/new_person.jpg --name "John Doe"
```

## 📌 Dependencies
-opencv-python

-face-recognition

-numpy

-pickle

## 👨‍💻 Author

**Developed by Allam Abdelmawgoud Ahmed 🚀 and hadeer bader**





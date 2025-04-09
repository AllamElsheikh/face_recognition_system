import os
import cv2
import face_recognition
import pickle
import argparse

# Function to train and save the face recognition model
def train_face_recognition_model(train_path, model_save_path="face_encodings.pkl"):
    def read_img(path):
        img = cv2.imread(path)

        if img is None:
            print(f"Error: Cannot load image {path}")
            return None  # Return None to skip this image

        if len(img.shape) == 2:  # Convert grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = img.astype("uint8")  # Ensure it's 8-bit

        print(f"Loaded image {path} with shape {img.shape} and dtype {img.dtype}")  # Debugging

        return img

    # Prepare known encodings and names
    known_encodings = []
    known_names = []
    known_ids = []
    known_dir = train_path

    # Read known images and extract their encodings
    for file in os.listdir(known_dir):
        img = read_img(os.path.join(known_dir, file))
        img_enc = face_recognition.face_encodings(img)
        if img_enc:  # Check if encodings were found
            known_encodings.append(img_enc[0])
            known_names.append(file.split('.')[0])  # Save the name without file extension
            known_ids.append(len(known_names))
    # Save the model (encodings and names) to a file
    with open(model_save_path, 'wb') as file:
        pickle.dump((known_encodings, known_names, known_ids), file)
    
    print(f"Model saved to {model_save_path}")



# Argument Parsing for command-line execution


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description="Train face recognition model")
  parser.add_argument("train_path" , help = "path to the training images directory.")
  parser.add_argument("--model_save_path" , default="face_encodings.pkl" , help = "path to save model")
  arg = parser.parse_args()

  train_face_recognition_model(arg.train_path, arg.model_save_path)


































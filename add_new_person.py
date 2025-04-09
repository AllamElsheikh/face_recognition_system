import os
import cv2
import face_recognition
import pickle
import argparse
from google.colab.patches import cv2_imshow

def read_img(path):
    """Load and resize an image to a fixed width of 500 pixels."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"❌ Error loading image: {path}")
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(img, (width, height))

def add_new_person(image_path, person_name, model_save_path="face_recognition_model.pkl"):
    """Add a new person to the face recognition model."""
    
    # Load the saved model (known_encodings and known_names)
    if os.path.exists(model_save_path):
        with open(model_save_path, 'rb') as file:
            known_encodings, known_names, known_ids = pickle.load(file)
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ No existing model found. Creating a new one.")
        known_encodings, known_names , known_ids = [], [] , []

    # Read and encode the new image
    img = read_img(image_path)
    img_encodings = face_recognition.face_encodings(img)

    # Check if a face was found in the image
    if img_encodings:
        img_encoding = img_encodings[0]  # Get the first face encoding
        known_encodings.append(img_encoding)  # Add encoding to the list
        known_names.append(person_name)       # Add name to the list
        known_ids.append(len(known_names))
        # Save the updated model
        with open(model_save_path, 'wb') as file:
            pickle.dump((known_encodings, known_names, known_ids), file)

        # Draw a bounding box around the detected face
        locations = face_recognition.face_locations(img)
        if locations:
            top, right, bottom, left = locations[0]
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, person_name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        # Display the image with the new person added
        cv2_imshow(img)
        print(f"✅ Successfully added {person_name} to the model.")
    else:
        print(f"❌ No faces found in the image {image_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new person to the face recognition model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the person's image")
    parser.add_argument("--name", type=str, required=True, help="Name of the person to add")
    parser.add_argument("--model_path", type=str, default="face_recognition_model.pkl", help="Path to save the model file")

    args = parser.parse_args()
    add_new_person(args.image_path, args.name, args.model_path)

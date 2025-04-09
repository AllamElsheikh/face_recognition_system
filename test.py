import os
import cv2
import face_recognition
import pickle
import argparse
import time
from google.colab.patches import cv2_imshow
from datetime import datetime



def read_img(path):
    """Load and resize an image to a fixed width of 500 pixels."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Error loading image: {path}")
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(img, (width, height))

def test_face_recognition_model(test_path, model_save_path, output_path , times =  "time_of_testing.pkl"):
    """Test a trained face recognition model and save recognized images."""
    
    # Load trained encodings and names
    with open(model_save_path, 'rb') as file:
        known_encodings, known_names, known_ids = pickle.load(file)
    
    print("✅ Model loaded successfully!")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    names = []
    ids = []
    timeings = []
    # Process images in test directory
    for file in os.listdir(test_path):
        img_path = os.path.join(test_path, file)
        print(f"🔍 Processing: {file}")

        try:
            img = read_img(img_path)
            img_enc = face_recognition.face_encodings(img)

            if img_enc:
                img_enc = img_enc[0]  # Extract the first face encoding
                results = face_recognition.compare_faces(known_encodings, img_enc)
                
                name = "Unknown"
                id = "unknown"
                for i, match in enumerate(results):
                    if match:
                        name = known_names[i]
                        id = known_ids[i]
                        names.append(name)  # Save the name without file extension
                        ids.append(id)
                        timeings.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                       
                        break
                
                # Draw a bounding box around the face
                locations = face_recognition.face_locations(img)
                if locations:
                    top, right, bottom, left = locations[0]
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(img, name, (left, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

                # Show image
                cv2_imshow(img)
                time.sleep(3)

                # Save recognized image
              #  start_time = time.time()

                output_img_path = os.path.join(output_path, f"recognized_{file}")
                cv2.imwrite(output_img_path, img)
                print(f"✅ Saved recognized image: {output_img_path}")
                print(f"the person is : {name}")
                print(f"the id is : {id}")
                names.append(name)  # Save the name without file extension
                ids.append(id)
                timeings.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                with open(times, 'wb') as file:
                  pickle.dump((names, ids, timeings), file)

            else:
                print(f"⚠️ No faces detected in {file}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained face recognition model on new images.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test images folder")
    parser.add_argument("--model_path", type=str, default="face_encodings.pkl", help="Path to the trained model file")
    parser.add_argument("--output_path", type=str, default="recognized_images", help="Path to save recognized images")

    args = parser.parse_args()
    test_face_recognition_model(args.test_path, args.model_path, args.output_path)

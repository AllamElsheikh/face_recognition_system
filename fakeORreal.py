from ultralytics import YOLO
import cv2

# Load the YOLO model once globally (to avoid loading it every time)
fake_real_model = YOLO("/content/face_recognition_system/best.pt")
classNames = ["fake", "real"]
confidence_threshold = 0.6

def predict_real_or_fake(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be read.")

    results = fake_real_model(img, stream=True, verbose=False)
    top_class = None
    top_conf = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf > confidence_threshold and conf > top_conf:
                top_conf = conf
                top_class = classNames[cls]
    print(top_class)
    return top_class  # Will return "real", "fake", or None
if __name__ == "__main__":
   predict_real_or_fake("/content/face_recognition_system/test/photo_2024-10-20_17-17-33.jpg")
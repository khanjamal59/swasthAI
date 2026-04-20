import numpy as np
import pickle
from PIL import Image
import cv2

# Optional: MediaPipe (Google Edge AI)
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False

# Load symptom columns
columns = pickle.load(open("ai_models/symptom_columns.pkl", "rb"))


# ---------------- SYMPTOM ENCODING ----------------
def encode_symptoms(symptom_string):
    symptoms = [s.strip().lower() for s in symptom_string.split(",")]

    vector = np.zeros(len(columns))

    for i, col in enumerate(columns):
        if col.lower() in symptoms:
            vector[i] = 1

    return vector.reshape(1, -1)


# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(file):
    try:
        # Convert file to OpenCV format
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image")

        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 🔥 MediaPipe enhancement (Edge AI)
        if MEDIAPIPE_AVAILABLE:
            with mp_face.FaceDetection(model_selection=0) as detector:
                results = detector.process(image_rgb)

                if results.detections:
                    h, w, _ = image.shape
                    bbox = results.detections[0].location_data.relative_bounding_box

                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    image = image[y:y+height, x:x+width]

        # Resize
        image = cv2.resize(image, (224, 224))

        # Normalize
        image = image.astype("float32") / 255.0

        # Expand dims
        image = np.expand_dims(image, axis=0)

        return image

    except Exception as e:
        print("❌ Image preprocessing error:", e)
        return None

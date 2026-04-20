import pickle
import numpy as np

# -------- LOAD SYMPTOM MODEL --------
try:
    symptom_model = pickle.load(open("ai_models/symptom_model.pkl", "rb"))
    print("✅ Symptom model loaded")

except Exception as e:
    print("❌ Symptom model error:", e)
    symptom_model = None


# -------- LOAD TFLITE MODEL (EDGE AI) --------
try:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path="ai_models/skin_model.tflite",
        num_threads=4   # 🔥 faster on CPU
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    TFLITE_AVAILABLE = True
    print("✅ TFLite model loaded successfully")

except Exception as e:
    print("❌ TFLite error:", e)

    interpreter = None
    input_details = None
    output_details = None
    TFLITE_AVAILABLE = False


# -------- DISEASE MAPPING --------
# 🔥 Make this dynamic if needed later
disease_map = {
    0: "Acne",
    1: "Eczema",
    2: "Infection"
}


# -------- OPTIONAL HELPER FUNCTION --------
def predict_image(image):
    """
    Runs inference on preprocessed image
    """
    if not TFLITE_AVAILABLE:
        return "Model not available", 0.0

    try:
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        pred_class = int(np.argmax(output))
        confidence = float(np.max(output))

        disease = disease_map.get(pred_class, "Unknown")

        return disease, confidence

    except Exception as e:
        print("❌ Prediction error:", e)
        return "Error", 0.0
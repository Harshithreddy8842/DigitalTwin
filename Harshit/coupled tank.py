import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained model (update path if needed)
model = load_model('tank_model.h5')

# Define state labels (update as per your training)
labels = ['Empty', 'Half', 'Full']

# Start video capture from default camera (laptop webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match model input size and normalize
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    x = np.expand_dims(img, axis=0)

    # Predict water level class
    prediction = model.predict(x)
    label_index = np.argmax(prediction)
    confidence = prediction[0][label_index]

    # Put prediction text on video frame
    text = f"{labels[label_index]} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video frame
    cv2.imshow('Coupled Tank Water Level', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

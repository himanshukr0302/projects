from keras.models import load_model
import cv2
import numpy as np

# Load face detector
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load your pre-trained emotion detection model (48x48 grayscale)
classifier = load_model('CNN_Model.keras')

# Emotion labelsq
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame, 'No Faces Detected', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)  # add channel dimension for grayscale: shape (48,48,1)
            roi = np.expand_dims(roi, axis=0)   # add batch dimension: shape (1,48,48,1)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            label_text = f"{label} ({confidence*100:.1f}%)"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Face ROI empty', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

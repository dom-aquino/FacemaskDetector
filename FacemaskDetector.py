import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

source = cv2.VideoCapture(0)

windowName = "Camera Preview"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

faceCascade = cv2.CascadeClassifier('classifiers/haarcascade_face_2.xml')
model = load_model("./mask_recog_v1.h5")

while cv2.waitKey(1) != 27:
    hasFrame, frame = source.read()

    if not hasFrame:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to greyscale
    greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(greyscale, scaleFactor=1.1, minNeighbors=4)

    faces_list = []
    preds = []
    # Draw rectangle across the detected face
    for (x, y, w, h) in face:
        face_frame = frame[y:y+h,x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (200, 200))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        preds = model.predict(face_frame)
        for pred in preds:
            withoutMask, mask = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y- 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display frame
    cv2.imshow(windowName, frame)

source.release()
cv2.destroyWindow(windowName)


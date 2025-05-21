import cv2
import os

# Load Haar cascades
fist_cascade = cv2.CascadeClassifier('fist.xml')
palm_cascade = cv2.CascadeClassifier('palm.xml')

# Check if cascades are loaded
if fist_cascade.empty() or palm_cascade.empty():
    print("Error loading cascade files")
    exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect fists
    fists = fist_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in fists:
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
          cv2.putText(frame, "Fist", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Detect palms
    palms = palm_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in palms:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Palm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Gesture Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  # Release resources
cap.release()
cv2.destroyAllWindows()

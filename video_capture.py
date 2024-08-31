import cv2
from simple_facerec import SimpleFacerec

# Initialize the facial recognition class
sfr = SimpleFacerec()
sfr.load_known_faces_from_db()

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Detect faces and emotions
    face_locations, face_ids, emotions = sfr.detect_known_faces(frame)

    # Draw face boxes and display labels
    for (top, right, bottom, left), name, emotion in zip(face_locations, face_ids, emotions):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        label = f"{name}: {emotion}"
        cv2.putText(frame, label, (left + 6, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

    # Display the video
    cv2.imshow("Video", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

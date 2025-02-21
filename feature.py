import cv2

# Initialize ORB detector
orb = cv2.ORB_create()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw keypoints on the frame
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

    # Show the frame
    cv2.imshow("ORB Feature Detection", frame_with_keypoints)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

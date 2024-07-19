import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize video capture
try:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)   # Set height
except Exception as e:
    print(f"Error initializing video capture: {e}")
    exit(1)

# Initialize the hand detector
try:
    detector = HandDetector(detectionCon=0.8)
except Exception as e:
    print(f"Error initializing hand detector: {e}")
    cap.release()
    exit(1)

# Initialize variables for the rectangle
rect_start = (100, 100)
rect_end = (300, 300)
dragging = False

# Calculate the width and height of the rectangle
rect_width = rect_end[0] - rect_start[0]
rect_height = rect_end[1] - rect_start[1]

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Error reading frame from camera")
            break
        
        img = cv2.flip(img, 1)  # Flip the image horizontally

        # Detect hands in the image
        try:
            hands, img = detector.findHands(img)
        except Exception as e:
            print(f"Error detecting hands: {e}")
            continue

        if hands:
            try:
                lmlist = hands[0]['lmList']  # Get the first hand detected's landmark list
                index_finger_tip = lmlist[8]  # Tip of the index finger

                # Check if the index finger is inside the rectangle
                if rect_start[0] < index_finger_tip[0] < rect_end[0] and rect_start[1] < index_finger_tip[1] < rect_end[1]:
                    if hands[0]['type'] == "Right":  # Ensure it's the right hand
                        dragging = True

                # If dragging, update the position of the rectangle
                if dragging:
                    new_x, new_y = index_finger_tip[0], index_finger_tip[1]
                    rect_start = (new_x - rect_width // 2, new_y - rect_height // 2)
                    rect_end = (new_x + rect_width // 2, new_y + rect_height // 2)

                # If the index finger is not inside the rectangle, stop dragging
                if not (rect_start[0] < index_finger_tip[0] < rect_end[0] and rect_start[1] < index_finger_tip[1] < rect_end[1]):
                    dragging = False
            except KeyError as e:
                print(f"Key error: {e}")
            except IndexError as e:
                print(f"Index error: {e}")

        # Draw the rectangle on the image
        try:
            cv2.rectangle(img, rect_start, rect_end, (255, 0, 255), 3)
        except Exception as e:
            print(f"Error drawing rectangle: {e}")

        # Display the image
        cv2.imshow("Image", img)

        


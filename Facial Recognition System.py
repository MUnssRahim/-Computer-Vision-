import cv2
import numpy as np
import face_recognition
import os

images = []
classnames = []

# Replace <YourUsername> with your actual username
path = r"C:\Users\HP\Desktop\img"

try:
    mylist = os.listdir(path)
    print("Files in the directory:", mylist)
except FileNotFoundError:
    print("The specified path was not found. Please check the path and try again.")
except NotADirectoryError:
    print("The specified path is not a directory. Please check the path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")

# Loop through each file in the directory
for cls in mylist:
    # Construct full file path
    file_path = os.path.join(path, cls)
    print(file_path)
    print(cls)
    
    # Read the image
    currentimg = cv2.imread(file_path)
    
    # Check if the image was read successfully
    if currentimg is not None:
       images.append(currentimg)
       classnames.append(os.path.splitext(cls)[0])
    else:
        print(f"Failed to load image: {file_path}")

# Print the class names
print("Class names:", classnames)

def findencoding(images):
    encodinglist = []
    for imag in images:
        imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(imag)
        if encodings:
            encodinglist.append(encodings[0])
        else:
            print("No face encodings found in the image.")
    return encodinglist

# Call the findencoding function
encoder = findencoding(images)

# Print the number of encodings
print("Number of encodings:", len(encoder))

cap = cv2.VideoCapture(0)
while True:
    success, imag = cap.read()
    if not success:
        print("Failed to grab frame.")
        break
    
    imags = cv2.resize(imag, (0, 0), None, 0.25, 0.25)
    imags = cv2.cvtColor(imags, cv2.COLOR_BGR2RGB)
    
    facescurrent = face_recognition.face_locations(imags)
    facecurrentencoder = face_recognition.face_encodings(imags, facescurrent)
    
    for encodingface, faceloc in zip(facecurrentencoder, facescurrent):
        matches = face_recognition.compare_faces(encoder, encodingface)
        distances = face_recognition.face_distance(encoder, encodingface)
        
        # Print the distances for debugging
        print("Distances:", distances)
        
        matchindex = np.argmin(distances)
        if matches[matchindex]:
            name = classnames[matchindex].upper()
            print(name)
            
            # Convert the face location to a box on the original frame
            top, right, bottom, left = faceloc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(imag, (left, top), (right, bottom), (0, 255, 0), 2)

            # Put the name of the recognized face on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(imag, name, (left, top - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Optionally display the frame
    cv2.imshow("Video Feed", imag)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows

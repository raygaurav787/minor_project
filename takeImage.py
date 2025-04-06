import csv
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time

# Define the path for the Haar cascade
xml_path = r"C:\Users\91884\Desktop\MINOR PROJECT\minn\Attendance-Management-system-using-face-recognition\haarcascade_frontalface_default.xml"

# Verify if the file exists
if not os.path.exists(xml_path):
    print(f"❌ Error: XML file not found at {xml_path}")
else:
    print(f"✅ XML file found at {xml_path}")

# Load the Haar Cascade
detector = cv2.CascadeClassifier(xml_path)

# Check if it's loaded correctly
if detector.empty():
    print("❌ Error: Failed to load Haar cascade classifier.")
else:
    print("✅ Haar cascade loaded successfully!")

#
# Function to capture face images
def TakeImage(l1, l2, haarcascade_path, trainimage_path, message, err_screen, text_to_speech):
    if not l1 and not l2:
        print("Please enter your Enrollment Number and Name.")
        return
    elif not l1:
        print("Please enter your Enrollment Number.")
        return
    elif not l2:
        print("Please enter your Name.")
        return

    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("❌ Error: Could not access the webcam.")
            #text_to_speech("Error: Could not access the webcam.")
            return

        Enrollment = l1.strip()
        Name = l2.strip()
        sampleNum = 0
        directory = f"{Enrollment}_{Name}"
        path = os.path.join(trainimage_path, directory)
        print(f"Saving to directory: {path}")
        os.makedirs(path, exist_ok=True)  # Ensure directory exists

        while True:
            ret, img = cam.read()
            if not ret:
                print("❌ Error: Failed to capture an image from the webcam.")
                text_to_speech("Error: Failed to capture an image.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

            print(f"✅ Faces detected: {len(faces)}")  # Print the number of faces found

            for (x, y, w, h) in faces:
                print("Face found. Attempting to save...")
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                img_path = os.path.join(path, f"{Name}_{Enrollment}_{sampleNum}.jpg")
                print(f"Images saved at: {img_path}")

                padding = 20 

                x1 = max(x - padding, 0)
                y1 = max(y - padding, 0)
                x2 = min(x + w + padding, gray.shape[1])
                y2 = min(y + h + padding, gray.shape[0])

                cv2.imwrite(img_path, gray[y1:y2, x1:x2])


                cv2.imshow("Frame", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elif sampleNum >= 50:  # Capture 50 images
                break

        cam.release()
        cv2.destroyAllWindows()

        # Save student details in CSV
        csv_path = "StudentDetails/studentdetails.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "a+", newline="") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([Enrollment, Name])

        res = f"✅ Images saved for ER No: {Enrollment}, Name: {Name}"
        print(res)
        message.configure(text=res)
        text_to_speech(res)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        text_to_speech("An error occurred while capturing images.")

import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import re


def run_presentation():
    # Constants
    WIDTH, HEIGHT = 960, 540
    GESTURE_THRESHOLD = 350
    BUTTON_DELAY = 10

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    # Folder path containing images
    folder_path = "presentation"

    # Initialize variables
    img_number = 0
    button_press = False
    button_counter = 0
    annotation_start = False
    annotations = [[]]
    annotation_number = -1

    # Initialize hand detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    # Function to sort filenames containing numbers
    def sort_key(filename):
        numbers = re.findall(r"\d+", filename)
        return int(numbers[0]) if numbers else 0

    # Get list of image paths sorted by filename
    image_paths = [
        i
        for i in sorted(os.listdir(folder_path), key=sort_key)
        if not i.startswith(".")
    ]

    # Main loop
    while True:
        # Read frame from video capture
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Load current image from folder
        path_full_image = os.path.join(folder_path, image_paths[img_number])
        img_current = cv2.imread(path_full_image)

        # Detect hands in the frame
        hands, img = detector.findHands(img, flipType=False)
        cv2.line(
            img, (0, GESTURE_THRESHOLD), (1280, GESTURE_THRESHOLD), (0, 255, 0), 20
        )

        # Handle hand gestures
        if hands and not button_press:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            cx, cy = hand["center"]
            lm_list = hand["lmList"]
            index_finger = lm_list[8][0], lm_list[8][1]
            x_val = int(np.interp(lm_list[8][0], [WIDTH // 2, WIDTH], [0, WIDTH]))
            index_finger = x_val, lm_list[8][1]

            # Show and draw pointer
            if fingers == [1, 1, 1, 0, 0]:
                cv2.circle(img_current, index_finger, 12, (0, 255, 0), cv2.FILLED)
            if fingers == [1, 1, 0, 0, 0]:
                if not annotation_start:
                    annotation_start = True
                    annotation_number += 1
                    annotations.append([])
                cv2.circle(img_current, index_finger, 12, (0, 255, 0), cv2.FILLED)
                annotations[annotation_number].append(index_finger)
            else:
                annotation_start = False

            # Handle button press
            if cy <= GESTURE_THRESHOLD:
                if fingers == [0, 0, 0, 0, 0]:  # Left gesture
                    button_press = True
                    annotations = [[]]
                    annotation_number = -1
                    annotation_start = False
                    if img_number > 0:
                        img_number -= 1
                if fingers == [1, 0, 0, 0, 0]:  # Right gesture
                    button_press = True
                    annotations = [[]]
                    annotation_number = -1
                    annotation_start = False
                    if img_number < len(image_paths) - 1:
                        img_number += 1
                if fingers == [1, 1, 1, 1, 0]:  # Four fingers gesture
                    if annotations:
                        annotations.pop(-1)
                        annotation_number -= 1
                        button_press = True

        # Handle button delay
        if button_press:
            button_counter += 1
            if button_counter > BUTTON_DELAY:
                button_counter = 0
                button_press = False

        # Draw annotations on the current image
        for i in range(len(annotations)):
            for j in range(len(annotations[i])):
                if j != 0:
                    cv2.line(
                        img_current,
                        annotations[i][j - 1],
                        annotations[i][j],
                        (200, 0, 0),
                        10,
                    )

        # Resize and display images
        img_small = cv2.resize(img, (WIDTH, HEIGHT))
        h, w, _ = img_current.shape
        img_current[0:HEIGHT, w - WIDTH : w] = img_small
        cv2.imshow("Image", img_current)

        # Exit condition
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


# Call the function to run the presentation
run_presentation()

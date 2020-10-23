import cv2
import numpy as np
import imutils
import math
import tensorflow as tf
from tensorflow import keras
import pygame


# global variables
background = None
blur_value = 41  # GaussianBlur parameter
bg_sub_threshold = 50
learning_rate = 0
is_bg_captured = 0
hand = None
pred = ''
action = ''
percent = 0

# False if we are creating dataset, True if dataset is created
is_dataset_created = True
frame_count, image_count, limit = 0, 0, 410
index = 4
folder_name = "test_set"

gestures = {
    0: "Peace",
    1: "Palm",
    2: "Fist",
    3: "Thumbs-up",
    4: "L"
}

model = keras.models.load_model("gesture_model.h5")


def segment(image, threshold=25):
    global background

    # calculate the threshold of the diff image so we get the foreground(the hand)
    thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]

    # image enhancing
    kernel = None
    eroded = cv2.erode(thresholded, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # find the contours in the thresholded image and select the biggest one (cv2.RETR_TREE, RETR_EXTERNAL)
    contours, _ = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if no contours have been detected return None
    if len(contours) == 0:
        return None
    else:
        # select the biggest contours based on contour area, which is the hand
        segmented = max(contours, key=cv2.contourArea)
        return thresholded, segmented


# TODO: Popravi koga prstite kje se vo tupanica da ne se gledaat 5 kruga
def find_fingertips(segmented, cpy, filter_value=20):
    hull = cv2.convexHull(segmented, returnPoints=False)
    defects = cv2.convexityDefects(segmented, hull)
    points = []

    if type(defects) != type(None):
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            end = tuple(segmented[e][0])

            points.append(end)
        filtered = filter_points(points, filter_value)

        filtered.sort(key=lambda point: point[1])
        for idx, pt in zip(range(5), filtered):
            cv2.circle(cpy, (pt[0] + 300, pt[1] + 80), 8, [255, 255, 0], -1)


def filter_points(points, filter_value):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i] and points[j] and euclidean_distance(points[i], points[j]) < filter_value:
                points[j] = None

    filtered = []
    for point in points:
        if point is not None:
            filtered.append(point)
    return filtered


def euclidean_distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def remove_background(frame):
    fgmask = bg_model.apply(frame, learningRate=learning_rate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=2)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def create_dataset(thresholded, gesture_name, img_count):
    img = cv2.resize(thresholded, (224, 224))
    img_name = f'C:/Users/anani/Desktop/DPNS/{folder_name}/{gesture_name}_{image_count}.jpg'
    cv2.imwrite(img_name, img)
    print(f"{img_name} Written")
    img_count += 1
    return img_count


# noinspection PyShadowingNames
def predict_gesture(image):
    img = np.array(image, dtype='float32')
    img /= 255
    pred_array = model.predict(img)
    print(f'Pred_array {pred_array}')
    result = gestures[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    percent = float("%0.2f" % (max(pred_array[0]) * 100))
    return result, percent


if __name__ == '__main__':
    # initialize weight for running average
    avg_weight = 0.5

    # get reference to camera
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 80, 300, 340, 590

    # initialize number of frames
    num_of_frames = 0

    # frame loop until cancelled with pressing 'q'
    while True:
        # get current frame
        grabbed, frame = camera.read()
        frame_count += 1

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so we don't get mirrored view
        frame = cv2.flip(frame, 1)

        # make a copy of the frame
        copy = frame.copy()

        # get the height and width of the frame
        height, width = frame.shape[:2]

        if is_bg_captured == 1:

            img = remove_background(frame)

            # get the ROI
            roi = img[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)

            hand = segment(gray)

            # check whether the hand is segmented
            if hand is not None:
                # if yes the take the thresholded and segmented region
                thresholded, segmented = hand

                # draw the hand region and display the thresholded image
                cv2.drawContours(copy, [segmented + (right, top)], -1, (0, 0, 255))
                
                # find_fingertips(segmented, copy)
                # for finger in fingers:
                #     cv2.circle(copy, (finger[0] + 300, finger[1] + 80), 8, [255, 255, 0], -1)

                cv2.imshow("Thresholded", thresholded)

                if not is_dataset_created and frame_count % 7 == 0:
                    image_count = create_dataset(thresholded, gestures[index], image_count)
                    if image_count == limit:
                        camera.release()
                        cv2.destroyAllWindows()

        # draw the segmented hand
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(copy, "Place region of the hand inside box after capturing the background",
                    (5, 50), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(copy, f'Prediction: {pred} ({percent}%)', (5, 100), font, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(copy, f'Action: {action}', (5, 130), font, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(copy, f'1.Peace: Play', (5, 180), font, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(copy, f'2.Palm: Pause', (5, 210), font, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(copy, f'3.Fist: Stop', (5, 240), font, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(copy, f'4.Thumbs-up: Volume-up', (5, 270), font, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(copy, f'5.L: Volume-down', (5, 300), font, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)

        cv2.rectangle(copy, (left, top), (right, bottom), (0, 255, 0), 2)

        # display the frame with segmented hand
        cv2.imshow("Video Feed", copy)

        # take the keypress of the user
        keypress = cv2.waitKey(10) & 0xFF

        # press b to capture the background
        if keypress == ord('b'):
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load("C:/Users/anani/Desktop/DPNS/DJ Khaled ft. Drake - POPSTAR (Official Audio).mp3")
            pygame.mixer.music.set_volume(0.5)
            pygame.mixer_music.play()
            bg_model = cv2.createBackgroundSubtractorMOG2(0, bg_sub_threshold, detectShadows=False)
            is_bg_captured = 1
            print(" Background Captured")

        # press 'r' to reset the background
        elif keypress == ord('r'):
            bg_model = None
            is_bg_captured = 0
            print(" Background Reset")

        # if space bar is pressed
        elif keypress == 32:
            # Tuka se pravi predviduvanjeto za gestot

            if is_dataset_created and is_bg_captured == 1 and hand is not None:
                target = np.stack((thresholded,) * 3, axis=-1)
                target = cv2.resize(target, (224, 224))
                target = target.reshape(1, 224, 224, 3)
                pred, percent = predict_gesture(target)

                if pred == 'Peace':
                    action = "Play"
                    pygame.mixer.music.unpause()

                elif pred == 'Palm':
                    action = "Pause"
                    pygame.mixer.music.pause()

                elif pred == 'Fist':
                    action = "Stop"
                    pygame.mixer.music.stop()

                elif pred == 'Thumbs-up':
                    action = "Volume-up"
                    volume = pygame.mixer.music.get_volume()
                    volume += 0.1
                    pygame.mixer.music.set_volume(volume)

                elif pred == 'L':
                    action = "Volume-down"
                    volume = pygame.mixer.music.get_volume()
                    volume -= 0.1
                    pygame.mixer.music.set_volume(volume)

        # if the user pressed 'q', then stop the loop
        elif keypress == ord("q"):
            break

    # turn off the camera and destroy the windows
    camera.release()
    cv2.destroyAllWindows()



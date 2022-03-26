import cv2
import sys
import numpy as np

def detector(filename):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()
    cap = cv2.VideoCapture(filename)

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter('output.mp4', fourcc, 30.0, (cap_width, cap_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        people, weights = hog.detectMultiScale(gray, winStride = (8, 8))
        people = np.array([[x, y, x + width, y + height] for (x, y, width, height) in people])

        for (x1, y1, x2, y2) in people:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Video", frame)
        output.write(frame)

        if cv2.waitKey(33) == 27: break

    cap.release()
    output.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    detector(filename = sys.argv[1])


import cv2
import numpy as np
import mediapipe as mp


class HeadTracker:

    def __init__(self, cameraNo=1, width=640, height=480, minDetConf=0.7, minTrackConf=0.7):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.cap = cv2.VideoCapture(cameraNo)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 800
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) # 600
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=minDetConf, min_tracking_confidence=minTrackConf)
        self.centerPointIdx = 1


    def DrawCenterPoint(self):
        radius = 3
        color = (0, 255, 0)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        if not (0 <= self.centerPointIdx < len(self.idx_to_coordinates)):
            raise ValueError(f'Landmark index is out of range. Invalid landmark #{self.centerPointIdx}')
        image = cv2.circle(self.frame, self.centerPoint, radius, color, thickness)
        image = cv2.putText(image, str(self.centerPoint[0]) + " x " + str(self.centerPoint[1]) , org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        return image, self.centerPoint


    def ReadCamera(self):
        if self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                print("Cannot read from camera, closing camera.")
                self.cap.release()
            else:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)


    def getDetections(self):
        self.frame.flags.writeable = False
        self.results = self.face_mesh.process(self.frame)
        self.frame.flags.writeable = True
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        self.image_rows, self.image_cols, _ = self.frame.shape
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.idx_to_coordinates = {}
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmark_px = self.mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                self.image_cols, self.image_rows)
                    if landmark_px:
                        self.idx_to_coordinates[idx] = landmark_px
            self.centerPoint = (self.idx_to_coordinates[self.centerPointIdx][0], self.idx_to_coordinates[self.centerPointIdx][1])
        else:
            # Need to handle this later
            pass
        return self.centerPoint


    def getFrameSize(self):
        return self.width, self.height
        

    def releaseCamera(self):
        self.cap.release()



def main():
    tracker = HeadTracker(1)
    while True:
        tracker.ReadCamera()
        point = tracker.getDetections()
        image, _ = tracker.DrawCenterPoint()
        cv2.imshow("Face center point", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    tracker.releaseCamera()


if __name__ == "__main__":
	main()
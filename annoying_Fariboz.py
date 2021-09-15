
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])
					   
FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_FACE_OVAL = frozenset([(10, 338), (338, 297), (297, 332), (332, 284),
                                (284, 251), (251, 389), (389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162), (162, 21), (21, 54),
                                (54, 103), (103, 67), (67, 109), (109, 10)])

FACEMESH_MOUTH = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), 
						   (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), 
						   (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])								

## Outer line
# FACEMESH_MOUTH = frozenset([
# 							(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
#                             (17, 314), (314, 405), (405, 321), (321, 375),
#                             (375, 291), 
# 						    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
#                             (267, 269), (269, 270), (270, 409), (409, 291)])

## Inner line
# FACEMESH_MOUTH = frozenset([(78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
#                             (14, 317), (317, 402), (402, 318), (318, 324),
#                             (324, 308), 
# 						    (78, 191), (191, 80), (80, 81), (81, 82),
#                             (82, 13), (13, 312), (312, 311), (311, 310),
#                             (310, 415), (415, 308)])		

FACEMESH_MOUTH_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
FACEMESH_LEFT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
FACEMESH_RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
						
# connections = FACEMESH_MOUTH
mouth_contour = FACEMESH_MOUTH_CONTOUR
righteye_contour = FACEMESH_RIGHT_EYE_CONTOUR
lefteye_contour = FACEMESH_LEFT_EYE_CONTOUR
between_left_and_right_eyes = [263, 33]

# mouth_center_pos = [380, 520]
# # left_eye_center_pos = [530, 350]
# # right_eye_center_pos = [270, 350]
# left_eye_center_pos = [270, 350]
# right_eye_center_pos = [530, 350]

mouth_center_pos = [325, 545]
left_eye_center_pos = [420, 340]
right_eye_center_pos = [230, 340]

mouth_width = 350
eyes_width = 150

def DrawOverlay(img, bound_idxs, coordinates):
	mask = np.zeros(img.shape[0:2], dtype=np.uint8)
	num_landmarks = len(coordinates)
	points = []
	# points = np.array([[[100,350],[120,400],[310,350],[360,200],[350,20],[25,120]]])
	if coordinates:
		for bound_idx in bound_idxs:
			if not (0 <= bound_idx < num_landmarks):
				raise ValueError(f'Landmark index is out of range. Invalid landmark #{bound_idx}')
			landmark_coordinate = coordinates[bound_idx]
			# if landmark_coordinate in coordinates:
			points.append([landmark_coordinate[0], landmark_coordinate[1]])
		points = np.asarray(points)
		# method 1 smooth region
		cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
		res = cv2.bitwise_and(img,img,mask = mask)
		rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
		cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
		## create the white background of the same size of original image
		wbg = np.ones_like(img, np.uint8)*255
		cv2.bitwise_not(wbg,wbg, mask=mask)
		# overlap the resulted cropped image on the white background
		dst = wbg+res
		# cv2.imshow('Original',img)
		# cv2.imshow("Mask",mask)
		# cv2.imshow("Cropped", cropped )
		# cv2.imshow("Same Size Black Image", res)
		# cv2.imshow("Same Size White Image", dst)
	return cropped

def DrawMouthCenter(image, bound_idxs, coordinates):
	radius = 3
	color = (0, 255, 0)
	thickness = 2
	num_landmarks = len(coordinates)
	if coordinates:
		for bound_idx in bound_idxs:
			if not (0 <= bound_idx < num_landmarks):
				raise ValueError(f'Landmark index is out of range. Invalid landmark #{bound_idx}')
			landmark_coordinate = coordinates[bound_idx]
			# if landmark_coordinate in coordinates:
			point = (landmark_coordinate[0], landmark_coordinate[1])
			image = cv2.circle(image, point, radius, color, thickness)
	return image, point

def overlay(_image, _facepart):
	threshold = 0
	binary = _facepart > threshold
	binary_mask = np.asarray(binary, dtype=np.uint8)
	binary_mask = cv2.bitwise_not(binary_mask)
	binary_mask = (binary_mask // 255) * 255

	_image = cv2.bitwise_and(_image, binary_mask)

	mapped_image = cv2.bitwise_xor(_image, _facepart)

	return mapped_image

def mapping(resized_orange, resized_mouth, resized_left_eye, resized_right_eye, mouth_dim, left_eye_dim, right_eye_dim):

	blacked_background_ = np.zeros(resized_orange.shape, dtype=np.uint8)

	mouthheight = int(mouth_dim[1]/2)
	mouthwidth = int(mouth_dim[0]/2)

	lefteyeheight = int(left_eye_dim[1]/2)
	lefteyewidth = int(left_eye_dim[0]/2)

	righteyeheight = int(right_eye_dim[1]/2)
	righteyewidth = int(right_eye_dim[0]/2)

	# ============================
	if (mouth_dim[1]%2 == 1) :
		mouth_height_offset = 1
	else:
		mouth_height_offset = 0

	if (mouth_dim[0]%2 == 1) :
		mouth_width_offset = 1
	else:
		mouth_width_offset = 0
	# ============================
	if (left_eye_dim[1]%2 == 1) :
		left_eye_height_offset = 1
	else:
		left_eye_height_offset = 0

	if (left_eye_dim[0]%2 == 1) :
		left_eye_width_offset = 1
	else:
		left_eye_width_offset = 0
	# ============================
	if (right_eye_dim[1]%2 == 1) :
		right_eye_height_offset = 1
	else:
		right_eye_height_offset = 0

	if (right_eye_dim[0]%2 == 1) :
		right_eye_width_offset = 1
	else:
		right_eye_width_offset = 0
	# ============================

	blacked_background_[ mouth_center_pos[1] - mouthheight : mouth_center_pos[1] + mouthheight + mouth_height_offset, 
	mouth_center_pos[0] - mouthwidth : mouth_center_pos[0] + mouthwidth + mouth_width_offset, :] = resized_mouth[:,:,:]

	blacked_background_[ left_eye_center_pos[1] - lefteyeheight : left_eye_center_pos[1] + lefteyeheight + left_eye_height_offset, 
	left_eye_center_pos[0] - lefteyewidth : left_eye_center_pos[0] + lefteyewidth + left_eye_width_offset, :] = resized_left_eye[:,:,:]

	blacked_background_[ right_eye_center_pos[1] - righteyeheight : right_eye_center_pos[1] + righteyeheight + right_eye_height_offset, 
	right_eye_center_pos[0] - righteyewidth : right_eye_center_pos[0] + righteyewidth + right_eye_width_offset, :] = resized_right_eye[:,:,:]

	annoying_orange = overlay(resized_orange, blacked_background_)

	# cv2.imshow("Mouth resized on black BG", blacked_background_ )
	# cv2.imshow("Mouth mask", annoying_orange )

	return annoying_orange

def init():
	# will define it later on, if it is needed.
	return 0

def main():

	drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
	cap = cv2.VideoCapture(1)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 800
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 600
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	print(width, height)

	with mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh:
		while cap.isOpened():
			success, image = cap.read()
			if not success:
				print("Cannot read from camera, please check.")
				break

			# Convert the BGR image to RGB.
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			results = face_mesh.process(image)

			# Draw the face mesh annotations on the image.
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			image_rows, image_cols, _ = image.shape

			if results.multi_face_landmarks:
				for face_landmarks in results.multi_face_landmarks:
					idx_to_coordinates = {}
					for idx, landmark in enumerate(face_landmarks.landmark):
						landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
													image_cols, image_rows)
						if landmark_px:
							idx_to_coordinates[idx] = landmark_px

				fucking_image, _ = DrawMouthCenter(image, [1], idx_to_coordinates)
				cv2.imshow("Res", fucking_image)
				if cv2.waitKey(1) & 0xFF == 27:
					break
	cap.release()

if __name__ == "__main__":
	main()
import numpy as np
import cv2
import dlib

predictor_path = "shape_predictor_5_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

detector = dlib.get_frontal_face_detector()

hat_img = cv2.imread("hat_cowboy.png", -1)

r, g, b, alpha_hat = cv2.split(hat_img)

rgb_hat = cv2.merge((r, g, b))

def add_hat(img):

	dets = detector(img, 1)

	if len(dets) > 0:
		d = dets[0]
		x, y, w, h = d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top()
		face_shape = predictor(img, d)
		point1 = face_shape.part(0)
		point2 = face_shape.part(2)
		eyes_center = ((point1.x + point2.x) / 2, (point1.y + point2.y) / 2)
		scaling_factor = (w / rgb_hat.shape[1]) * 1.5
		resized_hat_w = int(rgb_hat.shape[1] * scaling_factor)
		resized_hat_h = int(rgb_hat.shape[0] * scaling_factor)


		resized_hat = cv2.resize(rgb_hat, (resized_hat_w, resized_hat_h))
		hat_mask = cv2.resize(alpha_hat, (resized_hat_w, resized_hat_h))

		start_y = y - resized_hat_h
		if start_y < 0:
			resized_hat = resized_hat[-start_y:, :]
			hat_mask = hat_mask[-start_y:, :]
			start_y = 0
		end_y = y
		start_x = int(eyes_center[0] - resized_hat_w / 2)
		end_x = start_x + resized_hat_w
		if start_x < 0:
			resized_hat = resized_hat[:, -start_x:]
			hat_mask = hat_mask[:, -start_x:]
			start_x = 0
		elif end_x > img.shape[1]:
			resized_hat = resized_hat[:, :img.shape[1] - end_x]
			hat_mask = hat_mask[:, : img.shape[1] - end_x]
			end_x = img.shape[1]

		bg_roi = img[start_y:end_y, start_x:end_x].astype(float)

		
		mask_inv = cv2.bitwise_not(hat_mask)

		alpha = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR).astype(float) / 255
		
		bg = cv2.multiply(alpha, bg_roi).astype(int)
		hat = cv2.bitwise_and(resized_hat, resized_hat, mask=hat_mask)
		hat = cv2.resize(hat, (bg_roi.shape[1], bg_roi.shape[0])).astype(int)   
		hat = cv2.add(bg, hat)

		img[start_y:end_y, start_x:end_x] = hat
		return img
	return img


cap = cv2.VideoCapture(0)

while True:
	# Capture frame-by-frame
	ret, img = cap.read()
	output = add_hat(img)
	cv2.imshow("frame", output)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()

cv2.imwrite("output.png", output)
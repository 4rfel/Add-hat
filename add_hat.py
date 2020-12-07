import numpy as np 
import cv2
import dlib

def add_hat(img_ori, hat_img):

    img = np.copy(img_ori)

    r, g, b, a = cv2.split(hat_img) 
    rgb_hat = cv2.merge((r,g,b))

    predictor_path = "shape_predictor_5_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)  

    detector = dlib.get_frontal_face_detector()

    dets = detector(img, 1)

    if len(dets)>0:  
        for d in dets:
            x, y, w, h = d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top()
            shape = predictor(img, d)
            point1 = shape.part(0)
            point2 = shape.part(2)
            eyes_center = ((point1.x+point2.x)/2, (point1.y+point2.y)/2)
            scaling_factor = (w / rgb_hat.shape[1]) * 1.5
            resized_hat_w = int(rgb_hat.shape[1] * scaling_factor)
            resized_hat_h = int(rgb_hat.shape[0] * scaling_factor)

            resized_hat = cv2.resize(rgb_hat,(resized_hat_w,resized_hat_h))
            if resized_hat_h > y:
                resized_hat_h = y-1

            start_y = y - resized_hat_h
            end_y = y
            start_x = int(eyes_center[0] - resized_hat_w/2)
            end_x = int(eyes_center[0] + resized_hat_w/2)
            bg_roi = img[start_y: end_y, start_x: end_x].astype(float)

            hat_mask = cv2.resize(a, (resized_hat_w,resized_hat_h))
            mask_inv =  cv2.bitwise_not(hat_mask)
            
            mask_inv = cv2.merge((mask_inv, mask_inv, mask_inv))
            alpha = mask_inv.astype(float)/255
            alpha = cv2.resize(alpha, (bg_roi.shape[1], bg_roi.shape[0]))
            bg = cv2.multiply(alpha, bg_roi).astype(int)

            hat = cv2.bitwise_and(resized_hat, resized_hat, mask=hat_mask)
            hat = cv2.resize(hat, (bg_roi.shape[1], bg_roi.shape[0])).astype(int)
            add_hat = cv2.add(bg, hat)
            
            img[start_y:end_y, start_x: end_x] = add_hat
            return img


hat_img = cv2.imread("hat_cowboy.png",-1)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

output = add_hat(img, hat_img)
cv2.imwrite("output.png", output)

while 1:
    cv2.imshow('frame', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
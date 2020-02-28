import dlib
import cv2
import statistics
import numpy as np
cv2.destroyAllWindows()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_center(gray_img):
    moments = cv2.moments(gray_img, False)
    try:
        return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
    except:
        return None


def is_close(y0, y1):
    if abs(y0 - y1) < 5:
        return True
    return False


def matrix2line(arr):
    m = np.shape(arr)[0]
    n = np.shape(arr)[1]
    return arr.reshape(m*n,1)


def eye_point(img, parts, left=True):
    if left:
        eyes = [
                parts[36],
                min(parts[37], parts[38], key=lambda x: x.y),
                max(parts[40], parts[41], key=lambda x: x.y),
                parts[39],
                ]
    else:
        eyes = [
                parts[42],
                min(parts[43], parts[44], key=lambda x: x.y),
                max(parts[46], parts[47], key=lambda x: x.y),
                parts[45],
                ]
    if is_close(eyes[1].y, eyes[2].y):
        return None

    eye = img[eyes[1].y:eyes[2].y, eyes[0].x:eyes[3].x]
    #eye = img[eyes[0].x:eyes[3].x, eyes[1].y:eyes[2].y]
    imgGrayLineData = matrix2line(cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY))
    threshVal = np.median(imgGrayLineData)
    _, eye = cv2.threshold(cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY), threshVal, 255, cv2.THRESH_BINARY_INV)

    center = get_center(eye)
    if center:
        return center[0] + eyes[0].x, center[1] + eyes[1].y
    return center


def calc_black_whiteArea(bw_image):
    image_size = bw_image.size
    whitePixels = cv2.countNonZero(bw_image)
    blackPixels = bw_image.size - whitePixels
 
    whiteAreaRatio = (whitePixels/image_size)*100#[%]

    #blackAreaRatio = (blackPixels/image_size)*100#[%]
    ##print("White Area [%] : ", whiteAreaRatio)
    #print("Black Area [%] : ", blackAreaRatio)

    return whiteAreaRatio


def eye_ratio(img, parts, left=True):
    if left:
        eyes = [
                parts[36],
                min(parts[37], parts[38], key=lambda x: x.y),
                max(parts[40], parts[41], key=lambda x: x.y),
                parts[39],
                ]
    else:
        eyes = [
                parts[42],
                min(parts[43], parts[44], key=lambda x: x.y),
                max(parts[46], parts[47], key=lambda x: x.y),
                parts[45],
                ]
    if is_close(eyes[1].y, eyes[2].y):
        return None

    eye = img[eyes[1].y:eyes[2].y, eyes[0].x:eyes[3].x]
    eye_left_part = img[eyes[1].y:eyes[2].y, (eyes[3].x + eyes[0].x)//2:eyes[3].x]
    eye_right_part = img[eyes[1].y:eyes[2].y, eyes[0].x:(eyes[3].x + eyes[0].x)//2]

    imgGrayLineData = matrix2line(cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY))
    threshVal = np.median(imgGrayLineData)

    _, eye = cv2.threshold(cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY), threshVal, 255, cv2.THRESH_BINARY_INV)
    _, eye_left_part = cv2.threshold(cv2.cvtColor(eye_left_part, cv2.COLOR_RGB2GRAY), threshVal, 255, cv2.THRESH_BINARY_INV)
    _, eye_right_part = cv2.threshold(cv2.cvtColor(eye_right_part, cv2.COLOR_RGB2GRAY), threshVal, 255, cv2.THRESH_BINARY_INV)

    """
    if left==True:
        cv2.imshow("left_eye", eye)
    if left==False:
        cv2.imshow("right_eye", eye)
    """

    threshBW = 70
    if calc_black_whiteArea(eye_left_part) > threshBW:
        return 1
    elif calc_black_whiteArea(eye_right_part) > threshBW:
        return 2
    else:
        return 0


def p(img, parts, eye):
    if eye[0]:
        cv2.circle(img, eye[0], 3, (255, 255, 0), -1)
    if eye[1]:
        cv2.circle(img, eye[1], 3, (255, 255, 0), -1)

    for i in parts:
        cv2.circle(img, (i.x, i.y), 3, (255, 0, 0), -1)

    cv2.imshow("me", img)

cap = cv2.VideoCapture(0)

count = 1
while True:
    ret, frame = cap.read()

    #parts_pre = np.zeros(68)
    dets = detector(frame[:, :, ::-1])
    if len(dets) > 0:
        parts = predictor(frame, dets[0]).parts()
        #print(np.shape(parts))
        #print(parts[0])

        left_eye = eye_point(frame, parts)
        right_eye = eye_point(frame, parts, False)

        if count > 1:
            sum = 0
            for i in range(68):
                add = np.sqrt((parts_pre[i].x - parts[i].x) ** 2 + (parts_pre[i].y - parts[i].y) ** 2)
                sum = sum + add
            if sum > 100:
                moventFlag = 1
                print('movement flag is true')
            else:
                movementFlag = 0
        else:
            movementFlag = 1

        if movementFlag != 1:
            if eye_ratio(frame, parts) == 1 and eye_ratio(frame, parts, False) == 1:
                print('Left eye movement')
            elif eye_ratio(frame, parts) == 2 and eye_ratio(frame, parts, False) == 2:
                print('Right eye movement')
        """
        if left_eye != None and right_eye != None:
            #print(abs(left_eye[0] - parts[36].x))    # for debugging
            #print(abs(right_eye[0] - parts[42].x))   # for debugging
            threshEyeVal = 15
            if abs(left_eye[0] - parts[36].x) < threshEyeVal and abs(right_eye[0] - parts[42].x) < threshEyeVal:
                print('Right eye movement')
            if abs(left_eye[0] - parts[39].x) < threshEyeVal and abs(right_eye[0] - parts[45].x) < threshEyeVal:
                print('Left eye movement')
        """

        p(frame * 0, parts, (left_eye, right_eye))
        #p(frame, parts, (left_eye, right_eye))

        parts_pre = parts

        count = count + 1


    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
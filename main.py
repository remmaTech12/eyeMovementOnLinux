import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

STRAIGHT = 0
LEFT = 1
RIGHT = 2

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def get_center(gray_img):
    moments = cv2.moments(gray_img, False)
    try:
        return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
    except:
        return None

def matrix2line(arr):
    m = np.shape(arr)[0]
    n = np.shape(arr)[1]
    return arr.reshape(m*n,1)

def calc_white_area(bw_image):
    image_size = bw_image.size
    white_pixels = cv2.countNonZero(bw_image)
    white_area_ratio = (white_pixels/image_size)*100#[%]

    return white_area_ratio

def get_median_value_from_img(img):
    img_pixel_data = matrix2line(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    return np.median(img_pixel_data)

def is_eye_close(y0, y1):
    if abs(y0 - y1) < 10:
        return True
    return False

def get_eye_parts(parts, left):
    if left:
        eye_parts = [
                parts[36],
                min(parts[37], parts[38], key=lambda x: x.y),
                max(parts[40], parts[41], key=lambda x: x.y),
                parts[39],
                ]
    else:
        eye_parts = [
                parts[42],
                min(parts[43], parts[44], key=lambda x: x.y),
                max(parts[46], parts[47], key=lambda x: x.y),
                parts[45],
                ]

    if is_eye_close(eye_parts[1].y, eye_parts[2].y):
        return None
    else:
        return eye_parts

def get_eye_image(img, parts, left=True):
    eye_parts = get_eye_parts(parts, left)
    if eye_parts == None:
        return None, None

    eye_img = img[eye_parts[1].y:eye_parts[2].y, eye_parts[0].x:eye_parts[3].x]
    eye_img = mosaic(eye_img, 0.4)

    return eye_parts, eye_img

def get_eye_center(img, parts, left=True):
    eye_parts = get_eye_parts(parts, left)
    if eye_parts == None:
        return None

    eye_img = img[eye_parts[1].y:eye_parts[2].y, eye_parts[0].x:eye_parts[3].x]
    _, eye_img = cv2.threshold(cv2.cvtColor(eye_img, cv2.COLOR_RGB2GRAY), get_median_value_from_img(eye_img), 255, cv2.THRESH_BINARY_INV)

    center = get_center(eye_img)
    if center:
        return center[0] + eye_parts[0].x, center[1] + eye_parts[1].y
    else:
        return center

def eye_moving_direction(img, parts, left=True):
    eye_parts = get_eye_parts(parts, True)
    if eye_parts == None:
        return None

    eye_left_edge = eye_parts[3].x
    eye_right_edge = eye_parts[0].x

    eye_left_part = img[eye_parts[1].y:eye_parts[2].y, (2*eye_left_edge + eye_right_edge)//3:eye_left_edge]
    eye_right_part = img[eye_parts[1].y:eye_parts[2].y, eye_right_edge:(eye_left_edge + 2*eye_right_edge)//3]

    eye_img = img[eye_parts[1].y:eye_parts[2].y, eye_parts[0].x:eye_parts[3].x]
    _, eye_left_part = cv2.threshold(cv2.cvtColor(eye_left_part, cv2.COLOR_RGB2GRAY), get_median_value_from_img(eye_img), 255, cv2.THRESH_BINARY_INV)
    _, eye_right_part = cv2.threshold(cv2.cvtColor(eye_right_part, cv2.COLOR_RGB2GRAY), get_median_value_from_img(eye_img), 255, cv2.THRESH_BINARY_INV)

    black_eye_occupancy_rate_threshold = 80
    if calc_white_area(eye_left_part) > black_eye_occupancy_rate_threshold:
        return LEFT
    elif calc_white_area(eye_right_part) > black_eye_occupancy_rate_threshold:
        return RIGHT
    else:
        return STRAIGHT

def is_face_moving(parts_pre, parts):
    sum = 0
    for i in range(68):
        add = np.sqrt((parts_pre[i].x - parts[i].x) ** 2 + (parts_pre[i].y - parts[i].y) ** 2)
        sum = sum + add

    moving_threshold = 500
    if sum > moving_threshold:
        print('You are MOVING.')
        return True
    else:
        return False

def show_text(img, direction):
    img = cv2.rectangle(img,(0,0),(200,50),(255,255,255),-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_arr = ['STRAIGHT', 'LEFT', 'RIGHT']
    
    cv2.putText(img,text_arr[direction], (0,35), font, 1, (0,0,0), 2, cv2.LINE_AA, False)

def show_image(img, parts, eye, direction):
    for i in range(2):
        cv2.circle(img, eye[i], 3, (255, 255, 0), -1)

    for i in parts:
        cv2.circle(img, (i.x, i.y), 3, (255, 0, 0), -1)

    show_text(img, direction)

    left_eye_parts, left_eye_img = get_eye_image(frame, parts)
    if left_eye_parts != None:
        img[left_eye_parts[1].y:left_eye_parts[2].y, left_eye_parts[0].x:left_eye_parts[3].x] = left_eye_img

    right_eye_parts, right_eye_img = get_eye_image(frame, parts, False)
    if right_eye_parts != None:
        img[right_eye_parts[1].y:right_eye_parts[2].y, right_eye_parts[0].x:right_eye_parts[3].x] = right_eye_img

    cv2.imshow("me", img)

cap = cv2.VideoCapture(0)

count = 1
while True:
    ret, frame = cap.read()

    dets = detector(frame[:, :, ::-1])
    if len(dets) > 0:
        parts = predictor(frame, dets[0]).parts()

        direction = STRAIGHT
        if count > 1:
            face_moving = is_face_moving(parts_pre, parts)
        else:
            face_moving = False

        if face_moving != True:
            if eye_moving_direction(frame, parts) == LEFT and eye_moving_direction(frame, parts, False) == LEFT:
                direction = LEFT
                print('Eye movement to the LEFT')
            if eye_moving_direction(frame, parts) == RIGHT and eye_moving_direction(frame, parts, False) == RIGHT:
                direction = RIGHT
                print('Eye movement to the RIGHT')
            if eye_moving_direction(frame, parts) == RIGHT and eye_moving_direction(frame, parts, False) == LEFT:
                direction = STRAIGHT
                print('Eye CROSSING movement is detected!!')

        left_eye = get_eye_center(frame, parts)
        right_eye = get_eye_center(frame, parts, False)
        show_image(frame*0, parts, (left_eye, right_eye), direction)

        parts_pre = parts
        count = count + 1

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
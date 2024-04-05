import cv2
import mediapipe as mp
import time
import utils, math
import numpy as np


# Landmark điểm nhận được
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [
        (int(point.x * img_width), int(point.y * img_height))
        for point in results.multi_face_landmarks[0].landmark
    ]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


# Tính toán khoảng cách
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Tỉ lệ nhắm mắt
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    # cv2.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv2.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio


# Rút mắt xử lí
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)

    cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)
    eyes = cv2.bitwise_and(gray, gray, mask=mask)
    eyes[mask == 0] = 155
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    cropped_right = eyes[r_min_y:r_max_y, r_min_x:r_max_x]
    cropped_left = eyes[l_min_y:l_max_y, l_min_x:l_max_x]

    return cropped_right, cropped_left


# Lấy vị trí của mắt
def positionEstimator(cropped_eye):
    try:
        h, w = cropped_eye.shape

        # Xử lí lọc nhiễu để lấy ra các điểm mắt tập trung
        gaussain_blur = cv2.GaussianBlur(cropped_eye, (9, 9), 0)
        median_blur = cv2.medianBlur(gaussain_blur, 3)

        ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)
        piece = int(w / 3)

        # Chia làm 3 phần
        right_piece = threshed_eye[0:h, 0:piece]
        center_piece = threshed_eye[0:h, piece : piece + piece]
        left_piece = threshed_eye[0:h, piece + piece : w]
        eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

        return eye_position, color
    except Exception as e:
        # Exception
        print("An error occurred:", e)
        return None, None


# Đếm và trả kết quả
def pixelCounter(first_piece, second_piece, third_piece):
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    eye_parts = [right_part, center_part, left_part]

    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ""
    if max_index == 0:
        pos_eye = "RIGHT"
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 1:
        pos_eye = "CENTER"
        color = [utils.YELLOW, utils.PINK]
    elif max_index == 2:
        pos_eye = "LEFT"
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye = "Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


# Tính hệ số góc
def calulate_slope(up, down):
    if up[0] != down[0]:
        return (up[1] - down[1]) / (up[0] - down[0])
    return 0


# Tính góc cần quay
def calculate_angle(slope):
    a = np.arctan(slope) * 180 / np.pi
    if a > 0:
        return -(90 - abs(a))
    else:
        return 90 - a


# Tính kích thước mới sau khi xoay
def calculate_new_size(frame, angle):
    w_new = round(
        abs(frame.shape[1] * np.cos(np.deg2rad(angle)))
        + abs(frame.shape[0] * np.sin(np.deg2rad(angle)))
    )
    h_new = round(
        abs(frame.shape[1] * np.sin(np.deg2rad(angle)))
        + abs(frame.shape[0] * np.cos(np.deg2rad(angle)))
    )
    return w_new, h_new


# Dịch ảnh vào trung tâm
def translate_image_to_center(image, new_size):
    curr_size = (image.shape[1], image.shape[0])
    new_image = np.zeros((new_size[1], new_size[0]) + (3,), dtype=np.uint8)

    new_image[
        (new_size[1] - curr_size[1]) // 2 : (new_size[1] + curr_size[1]) // 2,
        (new_size[0] - curr_size[0]) // 2 : (new_size[0] + curr_size[0]) // 2,
    ] = image
    return new_image


# Xoay ảnh
def rotate_image(image, angle, new_size):
    frame = translate_image_to_center(image, new_size)
    new_center = (new_size[0] // 2, new_size[1] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(new_center, angle, 1.0)
    frame = cv2.warpAffine(frame, rotation_matrix, new_size)
    return frame


# Lấy vị trí điểm mới
def rotate_point(point, angle, center, new_size, old_size):
    new_point = (
        point[0] + (new_size[0] - old_size[0]) // 2,
        point[1] + (new_size[1] - old_size[1]) // 2,
    )
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])
    point_homogeneous = np.array([new_point[0], new_point[1], 1]).reshape((3, 1))
    rotated_point = np.dot(rotation_matrix, point_homogeneous)
    return round(rotated_point[0][0]), round(rotated_point[1][0])


# Lấy hướng mặt
def get_direct_face(face, eye):
    if face[0] != eye[0]:
        slope = calulate_slope(eye, face)
        if abs(slope) < 2:
            if slope < 0:
                return "LEFT"
            else:
                return "RIGHT"
        else:
            return "CENTER"

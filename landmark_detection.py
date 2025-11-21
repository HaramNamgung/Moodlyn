import cv2
import numpy as np
import mediapipe as mp

def detect_landmarks(image):
    """
    얼굴의 주요 랜드마크(눈, 코, 입, 눈썹)를 감지하는 함수
    
    Args:
        image: 입력 이미지 (BGR)
    
    Returns:
        landmarks: 랜드마크 좌표 딕셔너리
        face_detected: 얼굴 감지 여부
    """
    
    # MediaPipe 초기화
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    
    # BGR을 RGB로 수동 변환 (픽셀 단위)
    rgb_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # BGR(B, G, R) → RGB(R, G, B)
            rgb_image[i, j, 0] = image[i, j, 2]  # R
            rgb_image[i, j, 1] = image[i, j, 1]  # G
            rgb_image[i, j, 2] = image[i, j, 0]  # B
    
    # 얼굴 랜드마크 감지
    results = mp_face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        return None, False
    
    # 첫 번째 얼굴의 랜드마크만 사용
    face_landmarks = results.multi_face_landmarks[0]
    h, w = image.shape[:2]
    
    # 랜드마크를 좌표로 변환
    landmarks = {
        'left_eye': [],
        'right_eye': [],
        'nose': [],
        'mouth': [],
        'left_eyebrow': [],
        'right_eyebrow': []
    }
    
    left_eyebrow_indices = [159, 145, 144, 153, 154]
    right_eyebrow_indices = [386, 372, 371, 380, 381]
    left_eye_indices = [33, 160, 158, 133, 130, 131]
    right_eye_indices = [362, 385, 387, 263, 362, 133]
    nose_indices = [1, 2, 98, 326, 327]
    mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91]
    
    for idx, landmark in enumerate(face_landmarks.landmark):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        
        if idx in left_eyebrow_indices:
            landmarks['left_eyebrow'].append((x, y))
        elif idx in right_eyebrow_indices:
            landmarks['right_eyebrow'].append((x, y))
        elif idx in left_eye_indices:
            landmarks['left_eye'].append((x, y))
        elif idx in right_eye_indices:
            landmarks['right_eye'].append((x, y))
        elif idx in nose_indices:
            landmarks['nose'].append((x, y))
        elif idx in mouth_indices:
            landmarks['mouth'].append((x, y))
    
    mp_face_mesh.close()
    
    return landmarks, True


def draw_landmarks(image, landmarks):
    """
    감지된 랜드마크를 이미지에 그리는 함수
    
    Args:
        image: 입력 이미지
        landmarks: 감지된 랜드마크
    
    Returns:
        그려진 이미지
    """
    
    result = image.copy()
    
    # 각 부위별 색상 설정
    colors = {
        'left_eye': (0, 255, 0),
        'right_eye': (0, 255, 0),
        'nose': (255, 0, 0),
        'mouth': (0, 0, 255),
        'left_eyebrow': (255, 255, 0),
        'right_eyebrow': (255, 255, 0)
    }
    
    for key, points in landmarks.items():
        color = colors.get(key, (255, 255, 255))
        for x, y in points:
            cv2.circle(result, (x, y), 2, color, -1)
    
    return result
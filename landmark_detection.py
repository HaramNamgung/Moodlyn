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
    
    # MediaPipe 얼굴 랜드마크 인덱스 (정확한 포인트들)
    left_eyebrow_indices = [70, 63, 105, 66, 107]
    right_eyebrow_indices = [296, 334, 293, 300, 276]
    left_eye_indices = [33, 7, 163, 144, 145, 153]
    right_eye_indices = [362, 382, 381, 380, 374, 373]
    nose_indices = [1, 2, 5, 4, 19, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 166, 79, 294, 455]
    
    # MediaPipe 공식 입 랜드마크 인덱스들
    mouth_indices = [
        61,   # 0: 왼쪽 입꼬리 (MOUTH_LEFT)
        13,   # 1: 윗입술 중간-왼쪽
        82,   # 2: 윗입술 왼쪽
        18,   # 3: 윗입술 중앙 (MOUTH_TOP)
        312,  # 4: 윗입술 오른쪽  
        308,  # 5: 윗입술 중간-오른쪽
        291,  # 6: 오른쪽 입꼬리 (MOUTH_RIGHT)
        324,  # 7: 아랫입술 중간-오른쪽
        318,  # 8: 아랫입술 오른쪽
        17,   # 9: 아랫입술 중앙 (MOUTH_BOTTOM)
        87,   # 10: 아랫입술 왼쪽
        84,   # 11: 아랫입술 중간-왼쪽
    ]
    
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
        for i, (x, y) in enumerate(points):
            # 입 랜드마크는 더 크게 표시
            radius = 4 if key == 'mouth' else 2
            cv2.circle(result, (x, y), radius, color, -1)
            
            # 입 포인트에 번호 표시 (입꼬리 확인용)
            if key == 'mouth':
                cv2.putText(result, str(i), (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 입꼬리 포인트 특별 표시 (더 큰 원)
    if 'mouth' in landmarks and len(landmarks['mouth']) > 6:
        # 왼쪽 입꼬리 (인덱스 0)
        if len(landmarks['mouth']) > 0:
            x, y = landmarks['mouth'][0]
            cv2.circle(result, (x, y), 8, (0, 255, 255), 2)  # 노란 원
            cv2.putText(result, 'L', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 오른쪽 입꼬리 (인덱스 6)  
        if len(landmarks['mouth']) > 6:
            x, y = landmarks['mouth'][6]
            cv2.circle(result, (x, y), 8, (0, 255, 255), 2)  # 노란 원
            cv2.putText(result, 'R', (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return result
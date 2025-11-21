import numpy as np
import math

def calculate_distance(point1, point2):
    """
    두 점 사이의 거리 계산
    
    Args:
        point1: (x1, y1)
        point2: (x2, y2)
    
    Returns:
        거리값
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_angle(point1, point2, point3):
    """
    세 점으로 만든 각도 계산 (point2가 꼭짓점)
    
    Args:
        point1: 첫 번째 점
        point2: 꼭짓점
        point3: 세 번째 점
    
    Returns:
        각도 (도 단위)
    """
    # 벡터 계산
    v1 = (point1[0] - point2[0], point1[1] - point2[1])
    v2 = (point3[0] - point2[0], point3[1] - point2[1])
    
    # 내적 계산
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # 크기 계산
    magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    # 코사인 역함수로 각도 계산
    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = max(-1, min(1, cos_angle))  # -1 ~ 1 범위 제한
    
    angle = math.acos(cos_angle)
    return math.degrees(angle)


def analyze_mouth(landmarks):
    """
    입의 기하학적 특징 분석
    
    Args:
        landmarks: 감지된 랜드마크 딕셔너리
    
    Returns:
        입_높이, 입_너비, 입_개방도
    """
    if len(landmarks['mouth']) < 2:
        return 0, 0, 0
    
    mouth_points = landmarks['mouth']
    
    # 입의 높이 (위아래 거리)
    mouth_height = calculate_distance(mouth_points[1], mouth_points[7])  # 위 - 아래
    
    # 입의 너비 (좌우 거리)
    mouth_width = calculate_distance(mouth_points[0], mouth_points[6])  # 좌 - 우
    
    # 입 개방도 (높이 / 너비)
    mouth_openness = mouth_height / mouth_width if mouth_width != 0 else 0
    
    return mouth_height, mouth_width, mouth_openness


def analyze_eyebrows(landmarks):
    """
    눈썹의 기하학적 특징 분석
    
    Args:
        landmarks: 감지된 랜드마크 딕셔너리
    
    Returns:
        왼쪽_눈썹_높이, 오른쪽_눈썹_높이, 눈썹_각도
    """
    left_eyebrow = landmarks['left_eyebrow']
    right_eyebrow = landmarks['right_eyebrow']
    
    if len(left_eyebrow) < 2 or len(right_eyebrow) < 2:
        return 0, 0, 0
    
    # 왼쪽 눈썹 높이 (시작점 - 끝점의 Y 차이)
    left_eyebrow_height = abs(left_eyebrow[0][1] - left_eyebrow[-1][1])
    
    # 오른쪽 눈썹 높이
    right_eyebrow_height = abs(right_eyebrow[0][1] - right_eyebrow[-1][1])
    
    # 눈썹 각도 (왼쪽 눈썹의 시작점과 끝점으로 계산)
    if len(left_eyebrow) >= 2:
        eyebrow_angle = calculate_angle(left_eyebrow[0], left_eyebrow[2], left_eyebrow[-1])
    else:
        eyebrow_angle = 0
    
    return left_eyebrow_height, right_eyebrow_height, eyebrow_angle


def analyze_eyes(landmarks):
    """
    눈의 기하학적 특징 분석
    
    Args:
        landmarks: 감지된 랜드마크 딕셔너리
    
    Returns:
        왼쪽_눈_개방도, 오른쪽_눈_개방도
    """
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    # 충분한 포인트가 있는지 확인
    if len(left_eye) < 5 or len(right_eye) < 5:
        return 0, 0
    
    # 왼쪽 눈 높이 (위 - 아래)
    left_eye_height = calculate_distance(left_eye[1], left_eye[4])
    
    # 왼쪽 눈 너비 (좌 - 우)
    left_eye_width = calculate_distance(left_eye[0], left_eye[3])
    
    # 왼쪽 눈 개방도
    left_eye_openness = left_eye_height / left_eye_width if left_eye_width != 0 else 0
    
    # 오른쪽 눈 높이
    right_eye_height = calculate_distance(right_eye[1], right_eye[4])
    
    # 오른쪽 눈 너비
    right_eye_width = calculate_distance(right_eye[0], right_eye[3])
    
    # 오른쪽 눈 개방도
    right_eye_openness = right_eye_height / right_eye_width if right_eye_width != 0 else 0
    
    return left_eye_openness, right_eye_openness


def classify_emotion(landmarks):
    """
    기하학적 특징을 이용해 감정 분류
    
    Args:
        landmarks: 감지된 랜드마크 딕셔너리
    
    Returns:
        감정, 신뢰도, 점수
    """
    
    # 각 특징 계산
    mouth_height, mouth_width, mouth_openness = analyze_mouth(landmarks)
    left_eyebrow_h, right_eyebrow_h, eyebrow_angle = analyze_eyebrows(landmarks)
    left_eye_open, right_eye_open = analyze_eyes(landmarks)
    
    # 감정 판별 로직 (개선)
    scores = {
        'happy': 0,
        'sad': 0,
        'angry': 0,
        'neutral': 0,
        'surprised': 0
    }
    
    # 행복 (입 크게 벌어짐)
    if mouth_openness > 0.5:
        scores['happy'] += 5
    elif mouth_openness > 0.35:
        scores['happy'] += 2
    
    # 슬픔 (입 닫혀있고, 눈썹 각도 크거나 작음)
    if mouth_openness < 0.2:
        scores['sad'] += 4
    if eyebrow_angle > 130:  # 눈썹 내려옴
        scores['sad'] += 3
    if eyebrow_angle < 90:  # 눈썹 올라옴
        scores['sad'] += 2
    
    # 화남 (눈썹 찌푸림 - 각도 작음)
    if eyebrow_angle < 85:
        scores['angry'] += 5
    if mouth_openness < 0.15:
        scores['angry'] += 3
    
    # 놀람 (눈 크게 뜨고 입 벌어짐)
    if (left_eye_open > 0.45 and right_eye_open > 0.45) and mouth_openness > 0.45:
        scores['surprised'] += 5
    
    # 무표정 (특정 특징 없음)
    if 0.2 <= mouth_openness <= 0.35 and 100 <= eyebrow_angle <= 130:
        scores['neutral'] += 4
    
    # 가장 높은 점수의 감정 선택
    emotion = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = scores[emotion] / total_score if total_score > 0 else 0
    
    return emotion


def get_emotion_details(landmarks):
    """
    감정 분석 상세 정보 반환
    
    Args:
        landmarks: 감지된 랜드마크 딕셔너리
    
    Returns:
        감정 분석 딕셔너리
    """
    
    emotion, confidence, scores = classify_emotion(landmarks)
    mouth_h, mouth_w, mouth_open = analyze_mouth(landmarks)
    left_eyeb, right_eyeb, eyebrow_ang = analyze_eyebrows(landmarks)
    left_eye_o, right_eye_o = analyze_eyes(landmarks)
    
    return {
        'emotion': emotion,
        'confidence': round(confidence * 100, 2),
        'scores': scores,
        'mouth_openness': round(mouth_open, 3),
        'eyebrow_angle': round(eyebrow_ang, 2),
        'left_eye_openness': round(left_eye_o, 3),
        'right_eye_openness': round(right_eye_o, 3)
    }
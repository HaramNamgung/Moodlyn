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


def calculate_full_angle(point1, point2, point3):
    """
    세 점으로 만든 각도 계산 (0~360도, point2가 꼭짓점)
    외적을 사용해서 방향성을 고려한 각도 계산
    
    Args:
        point1: 첫 번째 점 (왼쪽 입꼬리)
        point2: 꼭짓점 (가운데)
        point3: 세 번째 점 (오른쪽 입꼬리)
    
    Returns:
        각도 (0~360도)
    """
    # 벡터 계산
    v1 = (point1[0] - point2[0], point1[1] - point2[1])
    v2 = (point3[0] - point2[0], point3[1] - point2[1])
    
    # 내적과 외적 계산
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    
    # atan2를 사용해서 -π ~ π 범위의 각도 계산
    angle = math.atan2(cross_product, dot_product)
    
    # 0 ~ 360도로 변환
    angle_degrees = math.degrees(angle)
    if angle_degrees < 0:
        angle_degrees += 360
    
    return angle_degrees


def analyze_mouth(landmarks):
    """
    입의 기하학적 특징 분석
    
    Args:
        landmarks: 감지된 랜드마크 딕셔너리
    
    Returns:
        입_높이, 입_너비, 입_개방도, 입꼬리_각도
    """
    if len(landmarks['mouth']) < 10:  # mouth_points[9]까지 사용하므로 최소 10개 필요
        return 0, 0, 0, 180
    
    mouth_points = landmarks['mouth']
    
    # 입의 높이 (위아래 거리)
    mouth_height = calculate_distance(mouth_points[1], mouth_points[7])  # 위 - 아래
    
    # 입의 너비 (좌우 거리)
    mouth_width = calculate_distance(mouth_points[0], mouth_points[6])  # 좌 - 우
    
    # 입 개방도 (높이 / 너비)
    mouth_openness = mouth_height / mouth_width if mouth_width != 0 else 0
    
    # 입꼬리 각도
    corner_angle = analyze_mouth_corner_angle(landmarks)
    
    return mouth_height, mouth_width, mouth_openness, corner_angle


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


def analyze_mouth_corner_angle(landmarks):
    """
    입꼬리 각도 분석 (웃음/찡그림 판별)
    입술 가운데 점에서 왼쪽 끝, 오른쪽 끝으로의 벡터 사이 각도 계산
    
    Args:
        landmarks: 감지된 랜드마크 딕셔너리
    
    Returns:
        입꼬리_각도 (작을수록 웃음, 클수록 찡그림)
    """
    if len(landmarks['mouth']) < 10:  # mouth_points[9]까지 사용하므로 최소 10개 필요
        return 180  # 기본값 (중립)
    
    mouth_points = landmarks['mouth']
    
    # 디버깅: 각 포인트 좌표와 사용하는 포인트들 확인
    print("=== 입 포인트 좌표 확인 ===")
    for i, point in enumerate(mouth_points):
        print(f"Point {i}: ({point[0]}, {point[1]})")
    
    # 특별히 3번, 10번, 7번 포인트 확인
    print("\n=== 주요 포인트 좌표 ===")
    if len(mouth_points) > 10:
        print(f"Point 3 (왼쪽 입꼬리): {mouth_points[3]}")
        print(f"Point 10: {mouth_points[10]}")
        print(f"Point 7 (오른쪽 입꼬리): {mouth_points[7]}")
    print("========================")
    
    # 입꼬리 각도 계산을 위한 포인트들
    # Point 10을 가운데로 해서 3번-10번-7번 각도 계산
    if len(mouth_points) > 10:
        left_corner = mouth_points[3]    # 왼쪽 입꼬리
        center = mouth_points[10]        # Point 10 (가운데)
        right_corner = mouth_points[7]   # 오른쪽 입꼬리
        
        print(f"\n=== 새로운 계산 (3-10-7) ===")
        print(f"왼쪽 입꼬리 (3번): {left_corner}")
        print(f"가운데 (10번): {center}")
        print(f"오른쪽 입꼬리 (7번): {right_corner}")
        
    else:
        return 180  # 기본값
    
    # Point 10을 기준으로 한 입꼬리 각도 계산
    corner_angle = calculate_full_angle(left_corner, center, right_corner)
    print(f"새로운 입꼬리 각도: {corner_angle}도")
    print("===============================")
    
    return corner_angle


def classify_emotion(landmarks):
    """
    기하학적 특징을 이용해 감정 분류
    
    Args:
        landmarks: 감지된 랜드마크 딕셔너리
    
    Returns:
        감정, 신뢰도, 점수
    """
    
    # 각 특징 계산
    mouth_height, mouth_width, mouth_openness, corner_angle = analyze_mouth(landmarks)
    left_eye_open, right_eye_open = analyze_eyes(landmarks)
    
    # 감정 판별 로직 (입꼬리 각도 활용)
    scores = {
        'happy': 0,
        'sad': 0,
        'angry': 0,
        'neutral': 0,
        'surprised': 0
    }
    
    # 행복 (입꼬리 올라감 - 각도 180도 미만, 입 벌어짐)
    if corner_angle < 180:  # 입꼬리 올라감 (웃음)
        scores['happy'] += 15
    if mouth_openness > 0.4:
        scores['happy'] += 4
    elif mouth_openness > 0.25:
        scores['happy'] += 2
    
    # 슬픔 (입꼬리 내려감 - 각도 180도 초과)
    if corner_angle > 180:  # 입꼬리 내려감 (슬픔)
        scores['sad'] += 15
    if mouth_openness < 0.15:
        scores['sad'] += 3
    
    # 화남 (입 굳게 닫음, 입꼬리 약간 내려감)
    if corner_angle > 185:  # 입꼬리 많이 내려감
        scores['angry'] += 4
    if mouth_openness < 0.1:
        scores['angry'] += 5
    
    # 놀람 (눈 크게 뜨고 입 벌어짐, BUT 입꼬리는 중립적이어야 함)
    if (left_eye_open > 0.45 and right_eye_open > 0.45) and mouth_openness > 0.45:
        # 입꼬리가 거의 중립적일 때만 놀람으로 판별 (175~185도)
        if 175 <= corner_angle <= 180:
            scores['surprised'] += 6
    if mouth_openness > 0.6 and 175 <= corner_angle <= 185:
        scores['surprised'] += 3
    
    # 무표정 (180도 근처, 적당한 입 개방도)
    if 175 <= corner_angle <= 185 and 0.15 <= mouth_openness <= 0.35:
        scores['neutral'] += 5
    
    # 가장 높은 점수의 감정 선택
    emotion = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = scores[emotion] / total_score if total_score > 0 else 0
    
    return emotion, scores, confidence


def get_emotion_details(landmarks):
    """
    감정 분석 상세 정보 반환
    
    Args:
        landmarks: 감지된 랜드마크 딕셔너리
    
    Returns:
        감정 분석 딕셔너리
    """
    
    emotion, scores, confidence = classify_emotion(landmarks)
    mouth_h, mouth_w, mouth_open, corner_angle = analyze_mouth(landmarks)
    left_eye_o, right_eye_o = analyze_eyes(landmarks)
    
    return {
        'emotion': emotion,
        'confidence': round(confidence * 100, 1),
        'scores': scores,
        'mouth_openness': round(mouth_open, 3),
        'corner_angle': round(corner_angle, 1),
        'left_eye_openness': round(left_eye_o, 3),
        'right_eye_openness': round(right_eye_o, 3)
    }
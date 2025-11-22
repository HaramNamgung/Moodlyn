import cv2
from preprocessing import preprocess_image
from landmark_detection import detect_landmarks, draw_landmarks
from emotion_logic import classify_emotion, get_emotion_details
def main():
    # 1. 이미지 불러오기
    image = cv2.imread("assets/sad1.jpg")

    # 2. 전처리
    processed = preprocess_image(image)

    # 3. 랜드마크 추출
    landmarks, face_detected = detect_landmarks(processed)

    # 4. 랜드마크 그리기 (시각화)
    if face_detected:
        landmark_image = draw_landmarks(processed.copy(), landmarks)
        # 이미지 출력
        cv2.imshow('Detected Landmarks', landmark_image)
        print("랜드마크가 감지되었습니다! 이미지 창을 확인하세요.")
        print("아무 키나 누르면 창이 닫힙니다.")
        cv2.waitKey(0)  # 키 입력 대기
        cv2.destroyAllWindows()  # 창 닫기
    else:
        print("⚠️ 얼굴이 감지되지 않았습니다!")
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 4. 감정 분석
    emotion, scores, confidence = classify_emotion(landmarks)
    emotion_details = get_emotion_details(landmarks)
    
    print("=" * 60)
    print("감정 분석 결과")
    print("=" * 60)
    print(f"최종 감정: {emotion_details['emotion']}")
    print(f"신뢰도: {emotion_details['confidence']}%")
    print()
    print("📊 각 감정별 점수:")
    for emotion_name, score in emotion_details['scores'].items():
        print(f"  {emotion_name:>10}: {score:>2}점")
    print()
    print("🔍 특징값 분석:")
    print(f"  입 개방도: {emotion_details['mouth_openness']}")
    print(f"  입꼬리 각도: {emotion_details['corner_angle']}도")
    print(f"  왼쪽 눈 개방도: {emotion_details['left_eye_openness']}")
    print(f"  오른쪽 눈 개방도: {emotion_details['right_eye_openness']}")
    print("=" * 60)

if __name__ == "__main__":
    main()

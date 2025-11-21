import cv2
from preprocessing import preprocess_image
from landmark_detection import detect_landmarks, draw_landmarks
from emotion_logic import classify_emotion
def main():
    # 1. 이미지 불러오기
    image = cv2.imread("assets/sad.jpg")

    # 2. 전처리
    processed = preprocess_image(image)

    # 3. 랜드마크 추출
    landmarks, face_detected = detect_landmarks(processed)

    # 4. 감정 분석
    emotion = classify_emotion(landmarks)
    print("Detected Emotion:", emotion)

if __name__ == "__main__":
    main()

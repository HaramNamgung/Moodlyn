import cv2
import numpy as np

def preprocess_image(image):
    """
    이미지를 전처리하는 함수
    
    Args:
        image: 입력 이미지 (BGR)
    
    Returns:
        contrast stretching이 적용된 이미지 (컬러 유지)
    """
    
    # 각 채널별로 contrast stretching 적용 (컬러 정보 유지)
    result = np.zeros_like(image, dtype=np.uint8)
    
    for channel in range(image.shape[2]):
        channel_data = image[:, :, channel]
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        
        # LUT 생성 (각 채널별로)
        LUT = np.array([0 if i < min_val else (255 if i > max_val else int((i - min_val) / (max_val - min_val) * 255)) for i in range(256)], dtype=np.uint8)
        
        if max_val - min_val != 0:
            for i in range(channel_data.shape[0]):
                for j in range(channel_data.shape[1]):
                    normalized_pixel = LUT[channel_data[i, j]]
                    result[i, j, channel] = int(normalized_pixel)
        else:
            result[:, :, channel] = channel_data.copy()
    
    return result
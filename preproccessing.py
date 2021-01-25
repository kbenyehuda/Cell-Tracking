import cv2

def preproccess(img):
    return cv2.resize(img.astype('uint8'), dsize=(1400, 1400), interpolation=cv2.INTER_LINEAR)

from flask import Flask, request, jsonify
import numpy as np
import cv2
from paddleocr import PaddleOCR
import imutils

app = Flask(__name__)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use bilateral filter to remove noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Apply edge detection
    edged = cv2.Canny(gray, 30, 200)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If our approximated contour has four points, we can assume it's a label
        if len(approx) == 4:
            screenCnt = approx
            break
    else:
        return image  # If no contour found, return original

    # Apply perspective transform to get top-down view
    pts = screenCnt.reshape(4, 2)
    rect = np.array(imutils.perspective.order_points(pts), dtype="float32")
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

@app.route('/extract-nutrition', methods=['POST'])
def extract_nutrition_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    in_memory_file = file.read()
    np_img = np.frombuffer(in_memory_file, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    processed_image = preprocess_image(image)

    # Run OCR
    result = ocr.ocr(processed_image, cls=True)

    # Extract text
    extracted_text = []
    for line in result:
        for box in line:
            text, confidence = box[1]
            extracted_text.append(text)

    return jsonify({'text': " ".join(extracted_text)})

@app.route('/')
def health():
    return 'Nutrition Label OCR API is running!'
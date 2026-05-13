
import cv2
import numpy as np

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred

def detect_edges(blurred, low=50, high=150):
    return cv2.Canny(blurred, low, high)

def apply_roi(image, original):
    height, width = image.shape[:2]
    vertices = np.array([[
        (int(width * 0.1),  height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9),  height)
    ]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(image, mask)

    roi_visual = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2RGB)
    cv2.polylines(roi_visual, vertices, isClosed=True,
                  color=(0, 255, 0), thickness=3)
    return masked, roi_visual

def detect_lines(masked_edges):
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=150
    )
    return lines if lines is not None else []

def average_lines(image, lines):
    height = image.shape[0]
    left_lines, right_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < -0.3:
            left_lines.append((slope, intercept))
        elif slope > 0.3:
            right_lines.append((slope, intercept))

    def make_line(slope, intercept):
        y1 = height
        y2 = int(height * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]

    averaged = []
    if left_lines:
        s, i = np.mean(left_lines, axis=0)
        averaged.append(make_line(s, i))
    if right_lines:
        s, i = np.mean(right_lines, axis=0)
        averaged.append(make_line(s, i))
    return averaged

def draw_lane_lines(image, lines, color=(0, 0, 255), thickness=8):
    canvas = np.zeros_like(image)
    for x1, y1, x2, y2 in lines:
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
    result = cv2.addWeighted(image, 0.8, canvas, 1.0, 0)
    return result, canvas   
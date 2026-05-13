
import cv2
import os
from pipeline import (preprocess, detect_edges, apply_roi,detect_lines, average_lines, draw_lane_lines)
from display import show_image, show_images_grid, save_image


IMAGE_PATH  = r"C:\Users\KIIT\Desktop\VS Code\Mini Project\ML_Projects\Car_Lane_detection\Car_Lane.jpg"   
VIDEO_PATH  = r"C:\Users\KIIT\Desktop\VS Code\Mini Project\ML_Projects\Car_Lane_detection\videoplayback.mp4"   
OUTPUT_DIR  = "output"
MODE        = "video"              # "image" or "video"


def process_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image: {path}")
        return

    gray, blurred   = preprocess(img)
    edges           = detect_edges(blurred)
    masked, roi_vis = apply_roi(edges, img)
    raw_lines       = detect_lines(masked)
    avg_lines       = average_lines(img, raw_lines)
    result, canvas  = draw_lane_lines(img.copy(), avg_lines)

    # Show all pipeline steps
    show_images_grid({
        "Original":         (cv2.cvtColor(img,     cv2.COLOR_BGR2RGB), None),
        "Grayscale":        (gray,                                      'gray'),
        "Canny Edges":      (edges,                                     'gray'),
        "ROI Mask":         (roi_vis,                                   None),
        "Masked Edges":     (masked,                                    'gray'),
        "Final Result":     (cv2.cvtColor(result,  cv2.COLOR_BGR2RGB), None),
    })

    save_image(os.path.join(OUTPUT_DIR, "result.jpg"), result)

def process_video(path):
    cap    = cv2.VideoCapture(path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "lane_output.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray, blurred   = preprocess(frame)
        edges           = detect_edges(blurred)
        masked, _       = apply_roi(edges, frame)
        raw_lines       = detect_lines(masked)
        avg_lines       = average_lines(frame, raw_lines)
        result, _       = draw_lane_lines(frame.copy(), avg_lines)

        out.write(result)
        frame_num += 1
        if frame_num % 30 == 0:
            print(f"  ⏳ {frame_num}/{total} frames processed...")

        
        cv2.imshow("Lane Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n Video saved → {out_path}")

##main function
if __name__ == "__main__":
    if MODE == "image":
        process_image(IMAGE_PATH)
    elif MODE == "video":
        process_video(VIDEO_PATH)
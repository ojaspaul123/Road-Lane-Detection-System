# 🚗 Road Lane Detection — Computer Vision Project

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Internship](https://img.shields.io/badge/Internship-Pinnacle%20Labs-orange?style=flat-square)

> A real-time road lane detection system built with Python and OpenCV. Detects and overlays left and right lane boundaries on dashcam images and video footage using classical computer vision techniques — Canny Edge Detection, ROI Masking, and Hough Line Transform.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [What It Does](#-what-it-does)
- [Who Uses It](#-who-uses-it)
- [Real-World Impact](#-real-world-impact)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [Getting Started](#-getting-started)
- [Challenges Faced & How I Solved Them](#-challenges-faced--how-i-solved-them)
- [Internship](#-internship)

---

## 📖 About the Project

This project was built as part of my **Data Science Internship at [Pinnacle Labs](https://pinnaclelabs.tech/internship/)**. The goal was to design and implement a computer vision pipeline that can detect lane lines on roads from both static images and live video streams — a core module in any autonomous driving or Advanced Driver Assistance System (ADAS).

---

## 🔍 What It Does

The system takes a road image or dashcam video as input and processes it through a multi-step pipeline to identify and draw the left and right lane lines on the road.

**Step-by-step process:**

1. **Preprocessing** — Converts the frame to grayscale and applies Gaussian Blur to reduce noise
2. **Edge Detection** — Applies Canny Edge Detection to highlight strong gradients (lane edges)
3. **ROI Masking** — Crops the frame to a trapezoidal region of interest (road ahead only) to eliminate irrelevant background
4. **Line Detection** — Uses Probabilistic Hough Line Transform to detect line segments within the masked region
5. **Line Averaging** — Separates lines into left and right by slope, then averages them into one clean line per side
6. **Overlay** — Draws the final lane lines back onto the original frame with a semi-transparent blend

**Supports two modes:**
- `image` — Processes a single photo and displays all pipeline steps in a grid
- `video` — Processes a dashcam video frame-by-frame and saves the output as an `.mp4` file

---

## 👥 Who Uses It

| User | Use Case |
|------|----------|
| **Autonomous Vehicle Engineers** | Core lane-keeping module for self-driving cars |
| **ADAS Developers** | Lane Departure Warning (LDW) systems in modern vehicles |
| **Robotics Researchers** | Path planning and navigation for mobile robots |
| **Traffic Analysts** | Monitoring lane discipline and road safety compliance |
| **Computer Vision Students** | Learning classical CV pipeline design end-to-end |
| **Dashcam Software Developers** | Enhancing dashcam footage with real-time annotations |

---

## 🌍 Real-World Impact

Lane detection is one of the most studied and impactful problems in computer vision. Here is why it matters:

**🚘 Autonomous Driving**
Lane detection is a foundational module in every Level 2+ autonomous vehicle. Systems like Tesla Autopilot, Waymo, and Mobileye all rely on accurate lane boundary estimation to keep vehicles centered and make safe steering decisions.

**⚠️ Road Safety**
According to the NHTSA, lane departure crashes account for nearly **50% of all fatal road accidents**. Systems built on lane detection can alert drivers before they unintentionally cross a lane, drastically reducing accidents.

**🗺️ HD Map Building**
Mapping platforms use lane detection at scale to build centimeter-accurate HD maps, which are essential for navigation and autonomous planning.

**🔬 Academic Research**
Lane detection benchmarks like TuSimple and CULane are actively researched, with deep learning methods (LaneNet, UFLD, CLRNet) building directly on the classical pipeline introduced in projects like this one.

**📱 Mobile & Edge AI**
Lightweight classical pipelines like this one (no GPU required) remain highly relevant for edge deployment in low-cost dashcams and embedded automotive ECUs where deep learning is too resource-heavy.

---

## 📁 Project Structure

```
Car_Lane_Detection/
│
├── Car_Lane.jpg          # Sample road image for testing
├── videoplayback.mp4     # Sample dashcam video for testing
│
├── main.py               # Entry point — switches between image/video mode
├── pipeline.py           # Core CV pipeline (preprocess → detect → draw)
├── display.py            # Visualization utilities (show, grid, save)
│
└── output/
    ├── result.jpg        # Output image (generated after image mode run)
    └── lane_output.mp4   # Output video (generated after video mode run)
```

---

## 🔧 Pipeline Overview

```
Input Frame
    │
    ▼
Grayscale + Gaussian Blur        ← preprocess()
    │
    ▼
Canny Edge Detection             ← detect_edges()
    │
    ▼
ROI Trapezoid Mask               ← apply_roi()
    │
    ▼
Hough Line Transform             ← detect_lines()
    │
    ▼
Left / Right Line Averaging      ← average_lines()
    │
    ▼
Overlay on Original Frame        ← draw_lane_lines()
    │
    ▼
Output (Image / Video)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python numpy matplotlib
```

### Run on Image

Open `main.py` and set:
```python
MODE       = "image"
IMAGE_PATH = "Car_Lane.jpg"
```
Then run:
```bash
python main.py
```

### Run on Video

Set:
```python
MODE       = "video"
VIDEO_PATH = "videoplayback.mp4"
```
Then run:
```bash
python main.py
```
Press `q` to stop the video window. Output is saved to `output/lane_output.mp4`.

---

## 🧩 Challenges Faced & How I Solved Them

### 1. 🔴 Noisy / Broken Lane Lines
**Problem:** Canny edges picked up road textures, cracks, and shadows as false lines, especially in the ROI zone, making the detected lines unstable and jumpy.

**Solution:** Tuned the Gaussian blur kernel `(5, 5)` to smooth out fine texture noise before edge detection. Also tightened the slope threshold filter in `average_lines()` — only lines with `|slope| > 0.3` are treated as lane candidates, filtering out near-horizontal noise.

---

### 2. 🔴 ROI Not Fitting the Frame
**Problem:** Using fixed pixel coordinates for the region of interest caused the mask to be completely wrong when the image resolution changed between the photo and video inputs.

**Solution:** Switched to **relative (percentage-based) coordinates** using `width * 0.1`, `height * 0.6`, etc., so the ROI trapezoid scales automatically with any resolution.

---

### 3. 🔴 Missing Lane on One Side
**Problem:** When the road curved, one of the lanes had very few detected segments, causing `average_lines()` to return only one line (or none), leading to an asymmetric or blank overlay.

**Solution:** Added a `None`-safe guard — if `left_lines` or `right_lines` is empty, that side is simply skipped rather than throwing an error. This keeps the overlay stable even when a lane briefly disappears from the ROI.

---

### 4. 🔴 Video Output Was Corrupt / Not Playing
**Problem:** The output `.mp4` file was being written but could not be played on Windows Media Player or VLC after the script ran.

**Solution:** Changed the `VideoWriter` codec from `XVID` to `mp4v` (`cv2.VideoWriter_fourcc(*'mp4v')`), which produces a proper H.264-compatible MP4 container that is universally playable.

---

### 5. 🔴 Hardcoded File Paths Breaking Portability
**Problem:** The absolute Windows paths in `main.py` (`C:\Users\KIIT\Desktop\...`) made the project non-portable — anyone else cloning the repo would immediately get a `FileNotFoundError`.

**Solution:** Updated the paths in the README to use simple relative filenames (`Car_Lane.jpg`, `videoplayback.mp4`) and documented how to set them up, so the project works from any directory on any OS.

---

## 🏢 Internship

This project was built as a task during my **Data Science Internship** at:

**[Pinnacle Labs](https://pinnaclelabs.tech/internship/)**
> Pinnacle Labs offers internship opportunities across Python Development, Data Science, Artificial Intelligence, Web Development, and more — with stipends, offer letters, and completion certificates.

---

## 📄 License

This project is open-source and free to use for educational and research purposes.

---

*Built with ❤️ using Python & OpenCV*

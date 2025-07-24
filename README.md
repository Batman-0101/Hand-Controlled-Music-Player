# 🖐️ Hand Volume Controller

Control your system’s volume seamlessly using hand gestures detected by your webcam. This project leverages computer vision and gesture recognition to map finger movements to volume levels in real time.

Built with OpenCV, MediaPipe, and Python, it delivers an intuitive and touch-free way to manage audio.

---

## 🚀 Features

- ✋ Real-time hand tracking with MediaPipe
- 🔊 Volume changes based on thumb-index finger distance
- 📉 Smooth interpolation of finger distance to system volume range
- 🖼️ Live display of landmarks, bounding box, and volume bar

---

## 🧠 How It Works

1. Your webcam captures video.
2. MediaPipe identifies 21 hand landmarks.
3. We measure the distance between the tip of your thumb and index finger.
4. That distance is mapped to your system volume range using linear interpolation.
5. The louder you get, the further apart your fingers go — nice and visual.

---

# 💡 Tips
- For best accuracy, use in well-lit conditions.
- Try holding your hand parallel to the webcam.
- Modify thresholds to match your volume preferences.
- Place your audio files in the songs/ directory. Please name each file using the following format:
```TrackName_AuthorFirstName_AuthorLastName.mp3 Example: Sunshine_John_Doe.mp3```
---

## ⚙️ Requirements

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

### Dependencies:

- opencv-python
- pygame
- mutagen
- numpy
- mediapipe

---

# ▶️ Running the Project

```bash
python src/main.py
```
Make sure your webcam is plugged in and accessible.

## Acknowledgments
This project builds upon concepts from a FreeCodeCamp.org tutorial on hand gesture volume control. The playback control features and modifications are original work.

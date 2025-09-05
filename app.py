import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Crowd Counting & Heatmap Dashboard", layout="wide")
st.title("👥 Crowd Counting & Heatmap from CCTV Video")

uploaded_file = st.file_uploader("Upload a CCTV video", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Sidebar options
    st.sidebar.header("Detection Settings")
    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    stride = st.sidebar.slider("Process every Nth frame", 1, 10, 2)

    # Load YOLO model
    model = YOLO("yolov8n.pt")  # Nano model for speed

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stframe = st.empty()
    counts = []
    timestamps = []

    all_points = []  # store centroids for heatmap
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % stride != 0:
            continue

        # Run YOLO detection
        results = model.predict(frame, conf=confidence, verbose=False)

        count = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    all_points.append([cy, cx])

        timestamp = frame_num / fps
        counts.append(count)
        timestamps.append(timestamp)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

    # Save CSV
    df = pd.DataFrame({"timestamp_sec": timestamps, "count": counts})
    csv_path = "crowd_counts.csv"
    df.to_csv(csv_path, index=False)
    st.download_button("📥 Download Counts CSV", data=open(csv_path, "rb"), file_name="crowd_counts.csv")

    # Plot count over time
    st.subheader("📊 People Count Over Time")
    fig, ax = plt.subplots()
    ax.plot(timestamps, counts, color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("People Count")
    st.pyplot(fig)

    # Generate heatmap
    if all_points:
        heatmap_data = np.zeros((frame_height, frame_width))
        for pt in all_points:
            heatmap_data[pt[0], pt[1]] += 1

        # Smooth heatmap
        heatmap_data = cv2.GaussianBlur(heatmap_data, (0, 0), sigmaX=25, sigmaY=25)

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap="hot", cbar=False)
        plt.axis("off")

        heatmap_path = "heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        st.subheader("🔥 Crowd Density Heatmap")
        st.image(heatmap_path, caption="Crowd Heatmap", use_container_width=True)
        st.download_button("📥 Download Heatmap", data=open(heatmap_path, "rb"), file_name="heatmap.png")

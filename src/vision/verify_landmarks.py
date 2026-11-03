"""
SecureKYC AI - MediaPipe 3D Landmark Index Verification Script
Extracts facial landmarks from real video frames and draws candidate points:
- Upper Inner Lip: #13 (Red)
- Lower Inner Lip: #14 (Green)
- Nose Tip: #1 (Blue)
- Chin: #152 (Yellow)
Saves annotated verification images into data/processed/ for visual confirmation.
"""

import sys
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_CHECKPOINT_DIR

# Candidate Landmark Indices
LANDMARK_UPPER_INNER_LIP = 13
LANDMARK_LOWER_INNER_LIP = 14
LANDMARK_NOSE_TIP = 1
LANDMARK_CHIN = 152


def get_face_landmarker():
    """Initializes MediaPipe FaceLandmarker or legacy FaceMesh."""
    # Check if legacy solutions API is available
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ), "solutions"

    # Modern mediapipe.tasks API (MediaPipe 1.0+)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    task_model_path = MODEL_CHECKPOINT_DIR / "face_landmarker.task"
    MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not task_model_path.exists():
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        print(f"[*] Downloading face_landmarker.task from {url}...")
        import ssl
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as response, open(task_model_path, "wb") as out_file:
            out_file.write(response.read())
        print("[✓] Downloaded face_landmarker.task successfully.")

    base_options = python.BaseOptions(model_asset_path=str(task_model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    return landmarker, "tasks"


def verify_landmarks_on_real_clip(video_path: Path, output_dir: Path, num_frames_to_save: int = 3):
    """Processes sample frames from a real video and saves annotated landmark images."""
    if not video_path.exists():
        raise FileNotFoundError(f"Sample video clip not found at: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    landmarker, api_type = get_face_landmarker()
    frame_idx = 0
    saved_count = 0

    print(f"[*] Processing video clip for landmark verification: {video_path.name} (Using MediaPipe '{api_type}' API)")

    while cap.isOpened() and saved_count < num_frames_to_save:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        if frame_idx % 10 != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        if api_type == "solutions":
            results = landmarker.process(rgb_frame)
            if not results.multi_face_landmarks:
                continue
            landmarks = results.multi_face_landmarks[0].landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect(mp_image)
            if not results.face_landmarks:
                continue
            landmarks = results.face_landmarks[0]

        # Extract 2D pixel positions
        p_upper = (int(landmarks[LANDMARK_UPPER_INNER_LIP].x * w), int(landmarks[LANDMARK_UPPER_INNER_LIP].y * h))
        p_lower = (int(landmarks[LANDMARK_LOWER_INNER_LIP].x * w), int(landmarks[LANDMARK_LOWER_INNER_LIP].y * h))
        p_nose = (int(landmarks[LANDMARK_NOSE_TIP].x * w), int(landmarks[LANDMARK_NOSE_TIP].y * h))
        p_chin = (int(landmarks[LANDMARK_CHIN].x * w), int(landmarks[LANDMARK_CHIN].y * h))

        annotated_frame = frame.copy()

        # Draw Landmark Circles
        cv2.circle(annotated_frame, p_upper, 4, (0, 0, 255), -1)   # Red: Upper Lip #13
        cv2.circle(annotated_frame, p_lower, 4, (0, 255, 0), -1)   # Green: Lower Lip #14
        cv2.circle(annotated_frame, p_nose, 5, (255, 0, 0), -1)    # Blue: Nose Tip #1
        cv2.circle(annotated_frame, p_chin, 5, (0, 255, 255), -1)  # Yellow: Chin #152

        # Draw Landmark Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame, "#13 Upper Lip", (p_upper[0] + 8, p_upper[1]), font, 0.45, (0, 0, 255), 1)
        cv2.putText(annotated_frame, "#14 Lower Lip", (p_lower[0] + 8, p_lower[1] + 12), font, 0.45, (0, 255, 0), 1)
        cv2.putText(annotated_frame, "#1 Nose", (p_nose[0] + 8, p_nose[1]), font, 0.45, (255, 0, 0), 1)
        cv2.putText(annotated_frame, "#152 Chin", (p_chin[0] + 8, p_chin[1]), font, 0.45, (0, 255, 255), 1)

        # Draw distance vector lines
        cv2.line(annotated_frame, p_upper, p_lower, (255, 255, 255), 1)
        cv2.line(annotated_frame, p_nose, p_chin, (0, 255, 255), 1)

        out_name = f"landmark_verification_frame_{saved_count + 1}.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), annotated_frame)
        print(f"[✓] Saved annotated frame {saved_count + 1} to {out_path}")

        # Compute D_lip distance for verification log
        p13_3d = np.array([landmarks[13].x * w, landmarks[13].y * h, landmarks[13].z * w])
        p14_3d = np.array([landmarks[14].x * w, landmarks[14].y * h, landmarks[14].z * w])
        p1_3d = np.array([landmarks[1].x * w, landmarks[1].y * h, landmarks[1].z * w])
        p152_3d = np.array([landmarks[152].x * w, landmarks[152].y * h, landmarks[152].z * w])

        d_lip = np.linalg.norm(p13_3d - p14_3d) / (np.linalg.norm(p1_3d - p152_3d) + 1e-6)
        print(f"    - Frame {saved_count + 1}: D_lip = {d_lip:.4f}")

        saved_count += 1

    cap.release()
    if hasattr(landmarker, "close"):
        landmarker.close()

    return saved_count


if __name__ == "__main__":
    sample_mp4 = RAW_DATA_DIR / "original_sequences" / "Celeb-real" / "id0_0000.mp4"
    if not sample_mp4.exists():
        mp4_list = list((RAW_DATA_DIR / "original_sequences").rglob("*.mp4"))
        if mp4_list:
            sample_mp4 = mp4_list[0]
        else:
            print("[!] No MP4 sample files found in data/raw/original_sequences/")
            sys.exit(1)

    verify_landmarks_on_real_clip(sample_mp4, PROCESSED_DATA_DIR)

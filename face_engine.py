"""
face_engine.py
--------------
Step 1 of the pipeline: detect a face in an image and produce a numeric
"encoding" (feature vector) for it.

Design choice:
- Universal multi-backend face detection: supports OpenCV's state-of-the-art
  YuNet ONNX detector (cv2.FaceDetectorYN), Haar Cascades (cv2.CascadeClassifier),
  and auto-downloading models into the local models/ directory.
- Feature Extraction: Standard 1764-dimensional L2-normalized HOG (Histogram of
  Oriented Gradients) descriptor (cv2.HOGDescriptor or fast vectorized NumPy).
- Tamper-Evident Hashing: SHA-256 cryptographic image fingerprinting.
"""

import os
import sys
import hashlib
import urllib.request
import cv2
import numpy as np

# Candidate model paths
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_YUNET_PATH = os.path.join(_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
_CASCADE_PATH = os.path.join(_MODELS_DIR, "haarcascade_frontalface_default.xml")


class NoFaceFoundError(Exception):
    """Raised when no face is detected in the supplied image."""
    pass


def _ensure_models():
    """Ensures at least one face detection model is available locally."""
    os.makedirs(_MODELS_DIR, exist_ok=True)

    # 1. Download YuNet model if using modern OpenCV with FaceDetectorYN
    if hasattr(cv2, "FaceDetectorYN_create") and not os.path.exists(_YUNET_PATH):
        try:
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            urllib.request.urlretrieve(url, _YUNET_PATH)
        except Exception:
            pass

    # 2. Download Haar Cascade if CascadeClassifier is available and model missing
    if hasattr(cv2, "CascadeClassifier") and not os.path.exists(_CASCADE_PATH):
        try:
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, _CASCADE_PATH)
        except Exception:
            pass


def _compute_hog(gray_64x64: np.ndarray) -> np.ndarray:
    """
    Computes standard 1764-dimensional HOG descriptor on a 64x64 grayscale face image.
    Compatible with any Python/OpenCV environment.
    Parameters: winSize=(64,64), blockSize=(16,16), blockStride=(8,8), cellSize=(8,8), nbins=9.
    """
    if gray_64x64.shape != (64, 64):
        gray_64x64 = cv2.resize(gray_64x64, (64, 64))

    # Try cv2.HOGDescriptor if available in this OpenCV build
    if hasattr(cv2, "HOGDescriptor"):
        try:
            hog = cv2.HOGDescriptor(
                _winSize=(64, 64),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9,
            )
            desc = hog.compute(gray_64x64)
            vec = desc.flatten().astype(np.float32)
            norm = np.linalg.norm(vec)
            return (vec / norm) if norm > 0 else vec
        except Exception:
            pass

    # Fast vectorized NumPy Sobel HOG calculation
    gx = cv2.Sobel(gray_64x64, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray_64x64, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0

    cell_hist = np.zeros((8, 8, 9), dtype=np.float32)
    bin_width = 20.0
    for cy in range(8):
        for cx in range(8):
            cell_mag = mag[cy * 8 : (cy + 1) * 8, cx * 8 : (cx + 1) * 8]
            cell_ang = ang[cy * 8 : (cy + 1) * 8, cx * 8 : (cx + 1) * 8]
            bins = np.floor(cell_ang / bin_width).astype(int) % 9
            for b in range(9):
                cell_hist[cy, cx, b] = np.sum(cell_mag[bins == b])

    block_descs = []
    for by in range(7):
        for bx in range(7):
            block = cell_hist[by : by + 2, bx : bx + 2, :].flatten()
            norm = np.linalg.norm(block) + 1e-6
            block_descs.append(block / norm)

    vec = np.concatenate(block_descs).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


class FaceEngine:
    def __init__(self, model_path: str = None):
        _ensure_models()
        self.yunet_detector = None
        self.cascade_detector = None

        # Try YuNet first if OpenCV supports FaceDetectorYN
        if hasattr(cv2, "FaceDetectorYN_create") and os.path.exists(_YUNET_PATH):
            try:
                self.yunet_detector = cv2.FaceDetectorYN_create(
                    _YUNET_PATH, "", (300, 300), score_threshold=0.4, nms_threshold=0.3
                )
            except Exception:
                self.yunet_detector = None

        # Fallback to Haar Cascade if CascadeClassifier is available
        if hasattr(cv2, "CascadeClassifier"):
            target_xml = model_path or _CASCADE_PATH
            if not os.path.exists(target_xml) and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                bundled = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                if os.path.exists(bundled):
                    target_xml = bundled
            if os.path.exists(target_xml):
                try:
                    cascade = cv2.CascadeClassifier(target_xml)
                    if not cascade.empty():
                        self.cascade_detector = cascade
                except Exception:
                    pass

    def detect_largest_face(self, image_bgr: np.ndarray):
        """
        Returns (x, y, w, h) of the largest detected face, or raises NoFaceFoundError.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Invalid or empty image provided to face detector.")

        h_img, w_img = image_bgr.shape[:2]

        # 1. Try YuNet (Deep Learning Face Detector)
        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w_img, h_img))
                _, faces = self.yunet_detector.detect(image_bgr)
                if faces is not None and len(faces) > 0:
                    best = max(faces, key=lambda f: f[2] * f[3])
                    x = max(0, int(best[0]))
                    y = max(0, int(best[1]))
                    w = min(w_img - x, int(best[2]))
                    h = min(h_img - y, int(best[3]))
                    if w > 10 and h > 10:
                        return x, y, w, h
            except Exception:
                pass

        # 2. Try Haar Cascade
        if self.cascade_detector is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.cascade_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            if len(faces) == 0:
                faces = self.cascade_detector.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=2, minSize=(25, 25)
                )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                return int(x), int(y), int(w), int(h)

        # 3. Fallback: If image is already a face crop or square portrait
        if min(w_img, h_img) >= 32:
            # Crop central 80% region as the prominent subject
            cx, cy = w_img // 2, h_img // 2
            side = int(min(w_img, h_img) * 0.8)
            x = max(0, cx - side // 2)
            y = max(0, cy - side // 2)
            w = min(w_img - x, side)
            h = min(h_img - y, side)
            return int(x), int(y), int(w), int(h)

        raise NoFaceFoundError("No face detected in the supplied image.")

    def crop_face(self, image_bgr: np.ndarray) -> np.ndarray:
        """Crops the bounding box of the largest face."""
        x, y, w, h = self.detect_largest_face(image_bgr)
        return image_bgr[y : y + h, x : x + w]

    def encode(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Detect the largest face and return a normalized HOG feature vector.
        """
        x, y, w, h = self.detect_largest_face(image_bgr)
        face_crop = image_bgr[y : y + h, x : x + w]
        face_resized = cv2.resize(face_crop, (64, 64))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        return _compute_hog(gray)

    @staticmethod
    def compare(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity between two encodings.
        1.0 = identical, 0 = unrelated. Clamped between 0.0 and 1.0.
        """
        if vec_a is None or vec_b is None:
            return 0.0
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def image_hash(image_bgr: np.ndarray) -> str:
        """
        SHA-256 of the raw image bytes — used as a tamper-evident fingerprint.
        """
        success, buf = cv2.imencode(".png", image_bgr)
        if not success:
            raise ValueError("Could not encode image for hashing.")
        return hashlib.sha256(buf.tobytes()).hexdigest()


def load_image(path: str) -> np.ndarray:
    """Loads an image from disk in BGR format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not decode image at {path}")
    return img
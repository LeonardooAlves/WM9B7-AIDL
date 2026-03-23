"""
Object Detection & Instance Segmentation — Streamlit App
=========================================================
Created using a LLM

This Streamlit app provides an interactive dashboard for object detection
and instance segmentation using pre-trained PyTorch models and the OAK-D camera.

Usage:
    pip install streamlit torch torchvision opencv-python-headless Pillow numpy depthai
    streamlit run object_detection_streamlit_app.py

For Google Colab (requires tunnelling — see instructions at bottom of this file):
    !pip install streamlit torch torchvision opencv-python-headless Pillow numpy pyngrok
    # Then run with ngrok tunnel (see COLAB_INSTRUCTIONS at end of file)
"""

import streamlit as st
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    ssd300_vgg16, SSD300_VGG16_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
)

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import requests
from io import BytesIO
from pathlib import Path
import time
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Object Detection & Instance Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).resolve().parent

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Sample images — local paths with URL fallbacks
SAMPLE_IMAGES = {
    "Cat": {
        "local": BASE_DIR / "images" / "cat.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    },
    "Camera": {
        "local": BASE_DIR / "images" / "camera.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Canon_EOS_5D_Mark_III.jpg/800px-Canon_EOS_5D_Mark_III.jpg",
    },
    "Submarine": {
        "local": BASE_DIR / "images" / "submarine.bmp",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/USS_Virginia_%28SSN-774%29.jpg/1200px-USS_Virginia_%28SSN-774%29.jpg",
    },
    "WMG Logo": {
        "local": BASE_DIR / "images" / "wmgLogo2.jpg",
    },
    "Street Scene": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/A_photo_of_a_busy_street.jpg/1200px-A_photo_of_a_busy_street.jpg",
    },
}

MODEL_REGISTRY = {
    "Faster R-CNN (ResNet50-FPN v2)": {
        "type": "detection",
        "description": "Two-stage detector. High accuracy, moderate speed.",
        "loader": lambda: fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    },
    "SSD300 (VGG16)": {
        "type": "detection",
        "description": "One-stage detector. Fast, good for real-time.",
        "loader": lambda: ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT),
    },
    "RetinaNet (ResNet50-FPN v2)": {
        "type": "detection",
        "description": "One-stage with Focal Loss. Great accuracy.",
        "loader": lambda: retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT),
    },
    "FCOS (ResNet50-FPN)": {
        "type": "detection",
        "description": "Anchor-free one-stage detector.",
        "loader": lambda: fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT),
    },
    "Mask R-CNN (ResNet50-FPN v2)": {
        "type": "instance_segmentation",
        "description": "Instance segmentation. Detects + pixel masks.",
        "loader": lambda: maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    },
}


# ============================================================================
# COLOUR UTILITIES
# ============================================================================
def generate_colours(n=91):
    colours = []
    for i in range(n):
        hue = i / n
        sat = 0.7 + 0.3 * ((i * 7) % 3) / 2
        val = 0.8 + 0.2 * ((i * 11) % 3) / 2
        rgb = hsv_to_rgb([hue, sat, val])
        colours.append(tuple(int(c * 255) for c in rgb))
    return colours

COLOURS = generate_colours(len(COCO_CLASSES))


# ============================================================================
# MODEL CACHING (Streamlit cache persists across reruns)
# ============================================================================
@st.cache_resource
def load_model(model_name):
    """Load and cache a detection model."""
    info = MODEL_REGISTRY[model_name]
    model = info["loader"]()
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================================
# INFERENCE
# ============================================================================
def run_inference(model, image_rgb):
    """Run inference on an RGB numpy image. Returns predictions dict and time."""
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image_rgb).to(DEVICE)

    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model([img_tensor])
        t1 = time.perf_counter()

    return outputs[0], (t1 - t0) * 1000


# ============================================================================
# VISUALISATION
# ============================================================================
def draw_detections(image, predictions, confidence_threshold=0.5,
                    show_masks=True, mask_alpha=0.45):
    """Draw boxes, labels, scores, and optionally masks on the image."""
    annotated = image.copy()
    summary = []

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    has_masks = 'masks' in predictions and show_masks

    if has_masks:
        masks = predictions['masks'].cpu().numpy()

    keep = scores >= confidence_threshold
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    if has_masks:
        masks = masks[keep]

    for i in range(len(boxes)):
        label_id = int(labels[i])
        score = float(scores[i])
        class_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else f'class_{label_id}'
        colour = COLOURS[label_id % len(COLOURS)]

        if has_masks:
            mask = masks[i, 0] > 0.5
            colour_mask = np.zeros_like(annotated)
            colour_mask[mask] = colour
            annotated = cv2.addWeighted(annotated, 1.0, colour_mask, mask_alpha, 0)

        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        text = f"{class_name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        summary.append({
            'Class': class_name, 'Confidence': f"{score:.3f}",
            'BBox': f"({x1},{y1},{x2},{y2})",
            'Area (px)': int((x2 - x1) * (y2 - y1)),
            'Has Mask': '✓' if has_masks else '—'
        })

    return annotated, summary


# ============================================================================
# IMAGE SOURCE FUNCTIONS
# ============================================================================
def load_local_image(image_path):
    """Load a local image file and return as RGB numpy array."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)

def load_image_from_url(url):
    """Download an image from a URL and return as RGB numpy array."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return np.array(img)

def load_sample_image(name):
    """
    Load a sample image by name. Tries local path first, falls back to URL.
    """
    entry = SAMPLE_IMAGES[name]
    # Try local path first
    if "local" in entry and entry["local"].exists():
        return load_local_image(entry["local"])
    # Fall back to URL
    elif "url" in entry:
        return load_image_from_url(entry["url"])
    else:
        raise FileNotFoundError(
            f"Sample image '{name}' not found locally at {entry.get('local', 'N/A')} "
            f"and no URL fallback provided."
        )

def capture_from_webcam(camera_index=0):
    """Capture a single frame from the default webcam."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam (index {camera_index})")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture frame from webcam")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def capture_from_oakd():
    """Capture a single frame from the OAK-D camera (DepthAI v3 API)."""
    try:
        import depthai as dai
    except ImportError:
        raise ImportError(
            "depthai not installed. Install with: pip install depthai\n"
            "Ensure OAK-D is connected via USB 3.0."
        )

    with dai.Pipeline() as pipeline:
        cam_rgb = pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A,
            sensorFps=30.0,
        )

        preview = cam_rgb.requestOutput(size=(640, 480))
        output_q = preview.createOutputQueue()

        pipeline.start()

        # Warm up: discard initial frames to allow auto-exposure to settle
        for _ in range(10):
            output_q.get()

        img_frame = output_q.get()
        frame = img_frame.getCvFrame()

        pipeline.stop()

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ============================================================================
# STREAMLIT APP UI
# ============================================================================
def main():
    # --- Header ---
    st.title("Object Detection & Instance Segmentation")
    st.markdown(
        "**MSc Applied Artificial Intelligence — University of Warwick**  \n Created using a LLM \n\n"
        f"Using device: `{DEVICE}` | PyTorch `{torch.__version__}`"
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Model Selection")
        model_name = st.selectbox(
            "Choose a model:",
            list(MODEL_REGISTRY.keys()),
            index=0
        )
        model_info = MODEL_REGISTRY[model_name]
        st.info(f"**Type:** {model_info['type']}  \n{model_info['description']}")

        st.subheader("Image Source")
        image_source = st.radio(
            "Choose image source:",
            ["Sample Image", "Upload File", "Image URL", "Webcam", "OAK-D Camera"]
        )

        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 0.05, 0.99, 0.50, 0.05
        )
        show_masks = st.checkbox(
            "Show Instance Masks (Mask R-CNN)", value=True
        )
        compare_all = st.checkbox(
            "Compare All Models", value=False
        )

        st.markdown("---")
        st.markdown(
            "### 📖 About\n"
            "This dashboard demonstrates pre-trained PyTorch models for:\n"
            "- **Object Detection** (bounding boxes)\n"
            "- **Instance Segmentation** (pixel masks)\n\n"
            "Models are trained on the **COCO dataset** (80 classes)."
        )

    # --- Image Acquisition ---
    image_rgb = None

    if image_source == "Sample Image":
        sample_name = st.selectbox("Select a sample image:", list(SAMPLE_IMAGES.keys()))
        with st.spinner(f"Loading '{sample_name}'..."):
            try:
                image_rgb = load_sample_image(sample_name)
            except Exception as e:
                st.error(f"Failed to load sample image: {e}")

    elif image_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        if uploaded_file is not None:
            image_rgb = np.array(Image.open(uploaded_file).convert('RGB'))

    elif image_source == "Image URL":
        url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
        if url:
            with st.spinner("Downloading image..."):
                try:
                    image_rgb = load_image_from_url(url)
                except Exception as e:
                    st.error(f"Failed to download image: {e}")

    elif image_source == "Webcam":
        if st.button("📷 Capture from Webcam"):
            with st.spinner("Capturing..."):
                try:
                    image_rgb = capture_from_webcam()
                except Exception as e:
                    st.error(f"Webcam error: {e}")

    elif image_source == "OAK-D Camera":
        if st.button("📸 Capture from OAK-D"):
            with st.spinner("Capturing from OAK-D..."):
                try:
                    image_rgb = capture_from_oakd()
                except Exception as e:
                    st.error(f"OAK-D error: {e}")

    # --- Run Detection ---
    if image_rgb is not None:
        if not compare_all:
            # Single model
            with st.spinner(f"Loading {model_name}..."):
                model = load_model(model_name)

            with st.spinner("Running inference..."):
                predictions, inf_time = run_inference(model, image_rgb)
                is_seg = model_info["type"] == "instance_segmentation"
                annotated, summary = draw_detections(
                    image_rgb, predictions, confidence_threshold,
                    show_masks=(show_masks and is_seg)
                )

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image_rgb, width="stretch")
            with col2:
                st.subheader(f"Detections — {model_name}")
                st.image(annotated, width="stretch")

            # Metrics
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Objects Detected", len(summary))
            mcol2.metric("Inference Time", f"{inf_time:.0f} ms")
            mcol3.metric("Model Type", model_info["type"].replace("_", " ").title())

            # Summary table
            if summary:
                st.subheader("📋 Detection Summary")
                df = pd.DataFrame(summary)
                st.dataframe(df, width="stretch", hide_index=True)

                # Class distribution chart
                if len(summary) > 1:
                    class_counts = df['Class'].value_counts()
                    st.subheader("📊 Detected Class Distribution")
                    st.bar_chart(class_counts)
            else:
                st.warning("No objects detected above the confidence threshold.")

        else:
            # Compare all models
            st.subheader("📊 Model Comparison")
            st.image(image_rgb, caption="Input Image", width=400)

            results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, (name, info) in enumerate(MODEL_REGISTRY.items()):
                status_text.text(f"Running {name}...")
                model = load_model(name)
                preds, inf_time = run_inference(model, image_rgb)
                is_seg = info["type"] == "instance_segmentation"
                annotated, summary = draw_detections(
                    image_rgb, preds, confidence_threshold,
                    show_masks=(show_masks and is_seg)
                )
                results[name] = {
                    'annotated': annotated,
                    'time_ms': inf_time,
                    'num_detections': len(summary),
                    'type': info["type"],
                    'summary': summary
                }
                progress_bar.progress((idx + 1) / len(MODEL_REGISTRY))

            status_text.text("Done!")

            # Display all results
            for name, r in results.items():
                with st.expander(
                    f"**{name}** — {r['num_detections']} detections, "
                    f"{r['time_ms']:.0f} ms", expanded=True
                ):
                    st.image(r['annotated'], width="stretch")
                    if r['summary']:
                        st.dataframe(pd.DataFrame(r['summary']),
                                     width="stretch", hide_index=True)

            # Comparison chart
            st.subheader("⏱️ Speed vs Detection Count")
            comp_data = pd.DataFrame({
                'Model': [n.split('(')[0].strip() for n in results],
                'Inference Time (ms)': [r['time_ms'] for r in results.values()],
                'Detections': [r['num_detections'] for r in results.values()],
            })

            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(comp_data.set_index('Model')['Inference Time (ms)'])
            with col2:
                st.bar_chart(comp_data.set_index('Model')['Detections'])

            st.dataframe(comp_data, width="stretch", hide_index=True)

    # --- Theory Section (expandable) ---
    with st.expander("📖 Theory: Object Detection"):
        st.markdown("""
### What is Object Detection?
Object detection answers **two questions** simultaneously:
1. **What** objects are in the image? → Classification
2. **Where** are they? → Localisation

Each detection consists of:
- **Bounding box**: $(x_{min}, y_{min}, x_{max}, y_{max})$
- **Class label**: e.g., "person", "car", "dog"
- **Confidence score**: $p \\in [0, 1]$

### Key Architectures

| Model | Type | Key Innovation |
|-------|------|---------------|
| **Faster R-CNN** | Two-stage | Region Proposal Network (RPN) |
| **SSD** | One-stage | Multi-scale feature maps |
| **RetinaNet** | One-stage | Focal Loss for class imbalance |
| **FCOS** | One-stage | Anchor-free, per-pixel prediction |
| **YOLO** | One-stage | Grid-based, single pass |
        """)

    with st.expander("📖 Theory: Instance Segmentation"):
        st.markdown("""
### What is Instance Segmentation?
Instance segmentation combines **object detection** with **semantic segmentation** to produce:
- A **bounding box** for each object
- A **pixel-level mask** for each individual instance

### Mask R-CNN
Extends Faster R-CNN by adding a **mask prediction branch**:
- **ROI Align** (bilinear interpolation, no quantisation)
- **Mask head**: Predicts a 28×28 binary mask per class per ROI
- **Loss**: $L = L_{cls} + L_{box} + L_{mask}$
        """)

    with st.expander("📖 Evaluation Metrics"):
        st.markdown("""
### IoU (Intersection over Union)
$$\\text{IoU} = \\frac{|A \\cap B|}{|A \\cup B|}$$

### Mean Average Precision (mAP)
$$\\text{mAP} = \\frac{1}{C} \\sum_{c=1}^{C} \\text{AP}_c$$

COCO primary metric: mAP@[0.5:0.05:0.95]
        """)

    with st.expander("📖 About the OAK-D Camera"):
        st.markdown("""
### Hardware Overview
The **OAK-D** (OpenCV AI Kit with Depth) by Luxonis features:
- Intel Movidius Myriad X VPU (4 TOPS)
- 12MP RGB camera (up to 4K)
- Stereo depth cameras (up to 20m range)
- USB 3.0 interface
- On-device neural network inference

### Two Approaches
| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| **On-device** | Run on Myriad X VPU | Fast, low power | Limited model support |
| **Host-side** | Stream to host + PyTorch | Any model | Needs host compute |
        """)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()


# ============================================================================
# COLAB INSTRUCTIONS
# ============================================================================
# To run this Streamlit app from Google Colab, you need a tunnel since
# Colab does not expose local ports. Here's how:
#
# OPTION 1: Using pyngrok (recommended)
# ----------------------------------------
# In a Colab cell, run:
#
#   !pip install streamlit pyngrok torch torchvision opencv-python-headless Pillow
#
#   # Write the app to a file (copy this script's full content)
#   %%writefile object_detection_streamlit_app.py
#   ... (paste this file's content) ...
#
#   # Start the tunnel
#   from pyngrok import ngrok
#   import subprocess, threading
#
#   # You need a free ngrok account and auth token from https://ngrok.com/
#   ngrok.set_auth_token("YOUR_NGROK_TOKEN")
#
#   # Start Streamlit in background
#   proc = subprocess.Popen(
#       ["streamlit", "run", "object_detection_streamlit_app.py",
#        "--server.port", "8501", "--server.headless", "true"]
#   )
#
#   # Create tunnel
#   public_url = ngrok.connect(8501)
#   print(f"\n🚀 Streamlit app is live at: {public_url}\n")
#
#
# OPTION 2: Using localtunnel (no account needed)
# --------------------------------------------------
#   !pip install streamlit torch torchvision opencv-python-headless Pillow
#   !npm install -g localtunnel
#
#   # Start Streamlit
#   !streamlit run object_detection_streamlit_app.py --server.port 8501 --server.headless true &
#
#   # In another cell:
#   !lt --port 8501
#
#
# NOTE: Webcam and OAK-D features will NOT work in Colab since there is
# no physical camera access. Use "Sample Image", "Upload", or "URL" sources.

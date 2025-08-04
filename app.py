# app.py (版本：图表文字英文化)

# 1. 基础和AI相关的库导入
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO 
import time
import os
import streamlit as st
import tempfile
from io import BytesIO

# 2. 导入您的自定义模型
try:
    from HY_FPN_model import FPN as MyCustomFPN
except ImportError:
    st.error("ERROR: Could not import your custom model. Please ensure `HY_FPN_model.py` and `HY_FPN_decoder.py` exist.")
    class MyCustomFPN: pass 

# 3. 定义模型文件常量
YOLO_MODEL_FILENAME = "YOLOv8.pt"
SEG_MODEL_FILENAME = "HY-FPN.pth" 

# --- 辅助函数 (保持不变) ---
@st.cache_resource
def load_detection_model(path):
    if not os.path.exists(path):
        st.error(f"YOLOv8 model file not found: {path}")
        return None
    try:
        model = YOLO(path)
        st.success(f"YOLOv8 detection model loaded successfully: {path}")
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {e}")
        return None

@st.cache_resource
def load_segmentation_model(path):
    if not os.path.exists(path):
        st.error(f"Custom FPN model file not found: {path}")
        return None
    try:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Loading custom FPN model to device: {DEVICE}")
        model = MyCustomFPN(encoder_name="resnet34", encoder_weights="imagenet", classes=1, in_channels=3)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        st.success(f"Custom FPN (resnet34) segmentation model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading custom FPN model: {e}")
        return None

def preprocess_for_segmentation(eye_crop, input_size=(256, 256)):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToPILImage(), T.Resize(input_size), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

def postprocess_segmentation(output_mask, original_crop_shape, seg_threshold):
    mask_numpy = output_mask.squeeze().cpu().detach().numpy()
    probabilities = 1 / (1 + np.exp(-mask_numpy))
    probabilities_resized = cv2.resize(probabilities, (original_crop_shape[1], original_crop_shape[0]), interpolation=cv2.INTER_LINEAR)
    binary_mask = (probabilities_resized > seg_threshold).astype(np.uint8) * 255
    return binary_mask, np.sum(binary_mask == 255)

def calculate_laplacian_variance(image):
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


# --- 主分析逻辑 ---
def run_analysis(video_path, yolo_model, seg_model, config):
    # (视频处理循环部分保持不变，为简洁省略)
    # ...
    with tempfile.TemporaryDirectory() as temp_dir:
        output_video_path = os.path.join(temp_dir, 'processed_video.mp4')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open the uploaded video file.")
            return None, None, None, None
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        results_data, max_observed_area, blink_count, was_blinking = [], 0, 0, False
        progress_bar = st.progress(0, text="Analyzing video...")
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            timestamp = (frame_count + 1) / fps
            display_frame = frame.copy()
            is_blurry = calculate_laplacian_variance(frame) < config['LAPLACIAN_VAR_THRESHOLD']
            if is_blurry:
                status_text = f"Frame: {frame_count+1} - Blurry"
                results_data.append({'frame': frame_count + 1, 'timestamp': timestamp, 'area': -1, 'is_blinking': None, 'status': 'blurry'})
            else:
                yolo_results = yolo_model.predict(frame, conf=config['YOLO_CONF_THRESHOLD'], classes=[0], verbose=False)
                eye_box = yolo_results[0].boxes[0].xyxy[0].cpu().numpy().astype(int) if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0 else None
                current_area, is_blinking_this_frame = 0, None
                if eye_box is not None:
                    x1, y1, x2, y2 = eye_box
                    eye_crop = frame[y1:y2, x1:x2]
                    if eye_crop.size > 0:
                        input_tensor = preprocess_for_segmentation(eye_crop)
                        with torch.no_grad():
                            seg_output = seg_model(input_tensor)
                        binary_mask, current_area = postprocess_segmentation(seg_output, eye_crop.shape[:2], config['SEG_THRESHOLD'])
                        if current_area > config['MIN_OPEN_AREA_FOR_REF']: max_observed_area = max(max_observed_area, current_area)
                        if max_observed_area > config['MIN_OPEN_AREA_FOR_REF']:
                            is_blinking_this_frame = current_area < (max_observed_area * config['BLINK_AREA_THRESHOLD_RATIO'])
                            if is_blinking_this_frame and not was_blinking: blink_count += 1
                            was_blinking = is_blinking_this_frame
                        else: was_blinking = False
                        if current_area > 0:
                            colored_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR); colored_mask[binary_mask == 255] = (255, 0, 255)
                            display_frame[y1:y2, x1:x2] = cv2.addWeighted(eye_crop, 0.7, colored_mask, 0.3, 0)
                        results_data.append({'frame': frame_count + 1, 'timestamp': timestamp, 'area': current_area, 'is_blinking': is_blinking_this_frame, 'status': 'processed'})
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else: results_data.append({'frame': frame_count + 1, 'timestamp': timestamp, 'area': -1, 'is_blinking': None, 'status': 'no_eye'})
                status_text = f"Frame: {frame_count+1} Area: {int(current_area)}"
                if is_blinking_this_frame is not None: status_text += f" Blink: {is_blinking_this_frame} (Total: {blink_count})"
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Max Area Ref: {int(max_observed_area)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            out_video.write(display_frame)
            progress_bar.progress((frame_count + 1) / total_frames, text=f"Analyzing... {frame_count+1}/{total_frames}")
        cap.release(); out_video.release(); progress_bar.empty()
        
        df = pd.DataFrame(results_data)
        true_max_area = df[df['area'] > 0]['area'].max()
        if pd.isna(true_max_area): true_max_area = 0
        processed_frames_df = df[df['status'] == 'processed']
        average_area = processed_frames_df['area'].mean() if not processed_frames_df.empty and processed_frames_df['area'].sum() > 0 else 0
        df.loc[:, 'normalized_area'] = df['area'].apply(lambda x: x / true_max_area if x >= 0 and true_max_area > 0 else (0 if x >= 0 else -1))
        valid_frames = df[(df['status'] == 'processed') & (df['is_blinking'].notna())]
        total_time_seconds = valid_frames['timestamp'].max() - valid_frames['timestamp'].min() if not valid_frames.empty else 0
        blink_frequency_hz = blink_count / total_time_seconds if total_time_seconds > 0 else 0
        stats = {
            'total_blinks': blink_count, 'analysis_duration': total_time_seconds,
            'blink_frequency_hz': blink_frequency_hz, 'blink_frequency_per_min': blink_frequency_hz * 60,
            'average_area': average_area, 'max_area': true_max_area
        }

        # --- CHANGED: English Plot Labels ---
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data = df[df['normalized_area'] >= 0]
        if not plot_data.empty:
            ax.plot(plot_data['timestamp'], plot_data['normalized_area'], label='Normalized Eye Fissure Area')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Normalized Area (relative to max)')
            ax.set_title('Normalized Eye Fissure Area Over Time')
            ax.set_ylim(bottom=-0.05, top=1.1)
            ax.legend()
            ax.grid(True)
        
        with open(output_video_path, 'rb') as f:
            video_bytes = f.read()
        return video_bytes, df, fig, stats


# --- Streamlit 网页界面 ---
st.set_page_config(layout="wide", page_title="Mouse Blink Analysis Platform")
st.title("👁️ Mouse Blink Analysis Platform")
st.markdown("""
This platform uses **YOLOv8** for eye detection and a custom **FPN (ResNet34 encoder)** for eyelid segmentation.
Upload a video file, adjust the analysis thresholds if needed, and click "Start Analysis".
""")

with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("1. Upload Video File")
    uploaded_file = st.file_uploader("Select a video file", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    st.subheader("2. Analysis Thresholds")
    yolo_conf_threshold = st.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
    seg_threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.5, 0.01)
    laplacian_var_threshold = st.number_input("Blur Detection Threshold", min_value=0, value=15)
    blink_area_threshold_ratio = st.slider("Blink Area Ratio Threshold", 0.0, 1.0, 0.1, 0.01)
    start_button = st.button("🚀 Start Analysis", disabled=(uploaded_file is None), type="primary")

if start_button:
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            video_path = tmpfile.name
        
        st.info(f"Video file uploaded: {uploaded_file.name}. Processing...")
        
        with st.spinner("Loading models... (This may take a moment on first run)"):
            yolo_model = load_detection_model(YOLO_MODEL_FILENAME)
            seg_model = load_segmentation_model(SEG_MODEL_FILENAME)

        if yolo_model and seg_model:
            config = {
                'YOLO_CONF_THRESHOLD': yolo_conf_threshold,
                'SEG_THRESHOLD': seg_threshold,
                'LAPLACIAN_VAR_THRESHOLD': laplacian_var_threshold,
                'BLINK_AREA_THRESHOLD_RATIO': blink_area_threshold_ratio,
                'MIN_OPEN_AREA_FOR_REF': 100,
            }
            video_bytes, results_df, results_fig, stats = run_analysis(video_path, yolo_model, seg_model, config)
            os.remove(video_path)

            if video_bytes:
                st.header("📊 Analysis Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Blinks", f"{stats['total_blinks']}")
                col2.metric("Blink Rate (per min)", f"{stats['blink_frequency_per_min']:.2f}")
                col3.metric("Average Fissure Area", f"{stats['average_area']:.2f} px")
                col4.metric("Analyzed Duration", f"{stats['analysis_duration']:.2f} s")
                
                st.subheader("Processed Video"); st.video(video_bytes)
                st.download_button("📥 Download Processed Video", video_bytes, "processed_video.mp4", "video/mp4")

                # --- CHANGED: English UI Text ---
                st.subheader("Normalized Eye Fissure Area Over Time"); st.pyplot(results_fig)
                buf = BytesIO(); results_fig.savefig(buf, format="png")
                st.download_button("📥 Download Plot", buf, "eye_area_plot.png", "image/png")
                
                st.subheader("Detailed Frame-by-Frame Data"); st.dataframe(results_df)
                csv = results_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 Download CSV Data", csv, "blink_analysis_results.csv", "text/csv")
    else:
        st.warning("Please upload a video file first.")
# app.py (最终版: 整合 session_state 以修复结果消失问题)

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
import sqlite3
from datetime import datetime
import math

# 页面设置，应在脚本的最顶端
st.set_page_config(layout="wide", page_title="Mouse Blink Analysis Platform")

# 尝试导入自定义模型，如果失败则优雅地报错
try:
    from ME_FPN_model import FPN as MyCustomFPN
except ImportError:
    st.error("ERROR: Could not import your custom model. Please ensure `ME_FPN_model.py` and `ME_FPN_decoder.py` exist.")
    # 定义一个占位符类，以避免后续代码因 MyCustomFPN 未定义而崩溃
    class MyCustomFPN: pass

# --- 全局常量 ---
YOLO_MODEL_FILENAME = "detection model.pt"
SEG_MODEL_FILENAME = "segmentation model.pth"
DB_FILE = "analysis_history.db"

# --- 数据库函数 ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            analysis_timestamp TEXT NOT NULL, 
            original_filename TEXT NOT NULL, 
            total_blinks INTEGER, 
            blink_frequency_per_min REAL, 
            average_normalized_area REAL, 
            avg_local_min_normalized_area REAL,
            analysis_duration_s REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_results_to_db(filename, stats):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO analysis_results (
            analysis_timestamp, original_filename, total_blinks, 
            blink_frequency_per_min, average_normalized_area, avg_local_min_normalized_area, analysis_duration_s
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        filename, 
        stats['total_blinks'], 
        stats['blink_frequency_per_min'], 
        stats.get('average_normalized_area'),
        stats.get('avg_local_min_normalized_area'),
        stats['analysis_duration_s']
    ))
    conn.commit()
    conn.close()

def load_results_from_db():
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    try: 
        return pd.read_sql_query("SELECT * FROM analysis_results ORDER BY id DESC", conn)
    except Exception: 
        return pd.DataFrame()
    finally: 
        conn.close()

# --- 模型加载与缓存 ---
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

# --- 图像处理函数 ---
def preprocess_for_segmentation(eye_crop, input_size=(256, 256)):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToPILImage(), 
        T.Resize(input_size), 
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

def postprocess_segmentation(output_mask, original_crop_shape, seg_threshold):
    mask_numpy = output_mask.squeeze().cpu().detach().numpy()
    probabilities = 1 / (1 + np.exp(-mask_numpy))
    probabilities_resized = cv2.resize(probabilities, (original_crop_shape[1], original_crop_shape[0]), interpolation=cv2.INTER_LINEAR)
    binary_mask = (probabilities_resized > seg_threshold).astype(np.uint8)
    return binary_mask, np.sum(binary_mask)

def calculate_laplacian_variance(image): 
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

# --- 核心分析逻辑 ---
def run_analysis(video_path, yolo_model, seg_model, config):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_video_path = os.path.join(temp_dir, 'processed_video.mp4')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open the uploaded video file.")
            return None, None, None, None
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        frame_data_list = []
        max_observed_area = 0
        previous_area = -1
        blink_count = 0
        blink_state = 'OPEN' 
        blink_minima_areas = []
        current_blink_minimum = float('inf')
        local_minima_areas = []
        dip_state = 'IDLE'
        current_dip_minimum = float('inf')

        progress_bar = st.progress(0, text="Analyzing video...")
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            timestamp = (frame_count + 1) / fps
            display_frame = frame.copy()
            is_blurry = calculate_laplacian_variance(frame) < config['LAPLACIAN_VAR_THRESHOLD']
            status_text = ""
            current_area_for_display = 0

            if is_blurry:
                frame_data_list.append({'frame': frame_count + 1, 'timestamp': timestamp, 'area': -1, 'status': 'blurry'})
                status_text = f"Frame: {frame_count+1} Status: BLURRY"
            else:
                yolo_results = yolo_model.predict(frame, conf=config['YOLO_CONF_THRESHOLD'], classes=[0], verbose=False)
                eye_box = yolo_results[0].boxes[0].xyxy[0].cpu().numpy().astype(int) if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0 else None
                current_area = 0
                status = 'no_eye'
                if eye_box is not None:
                    x1, y1, x2, y2 = eye_box
                    eye_crop = frame[y1:y2, x1:x2]
                    if eye_crop.size > 0:
                        input_tensor = preprocess_for_segmentation(eye_crop)
                        with torch.no_grad(): seg_output = seg_model(input_tensor)
                        binary_mask, current_area = postprocess_segmentation(seg_output, eye_crop.shape[:2], config['SEG_THRESHOLD'])
                        status = 'processed'
                        if current_area > config['MIN_OPEN_AREA_FOR_REF']: max_observed_area = max(max_observed_area, current_area)
                        if previous_area != -1 and max_observed_area > config['MIN_OPEN_AREA_FOR_REF']:
                            area_drop = previous_area - current_area
                            area_rise = current_area - previous_area 
                            if blink_state == 'OPEN':
                                if area_drop > config['BLINK_DROP_THRESHOLD']: blink_state = 'CLOSING'; current_blink_minimum = previous_area
                            elif blink_state == 'CLOSING':
                                current_blink_minimum = min(current_blink_minimum, current_area)
                                if current_area < (max_observed_area * config['BLINK_CONFIRM_RATIO']): blink_state = 'CLOSED_CONFIRMED'
                                elif area_rise > 0: blink_state = 'OPEN'
                            elif blink_state == 'CLOSED_CONFIRMED':
                                current_blink_minimum = min(current_blink_minimum, current_area)
                                if area_rise > 0: blink_count += 1; blink_minima_areas.append(current_blink_minimum); blink_state = 'OPENING'
                            elif blink_state == 'OPENING':
                                if current_area > (max_observed_area * config['REOPEN_RATIO_THRESHOLD']): blink_state = 'OPEN'
                            if dip_state == 'IDLE':
                                if area_drop > config['DIP_THRESHOLD']: dip_state = 'DIPPING'; current_dip_minimum = current_area
                            elif dip_state == 'DIPPING':
                                current_dip_minimum = min(current_dip_minimum, current_area)
                                if area_rise > 0: local_minima_areas.append(current_dip_minimum); dip_state = 'IDLE'
                        previous_area = current_area
                        current_area_for_display = current_area
                        if current_area > 0:
                            colored_mask = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR); colored_mask[binary_mask == 1] = (255, 0, 255)
                            display_frame[y1:y2, x1:x2] = cv2.addWeighted(eye_crop, 0.7, colored_mask, 0.3, 0)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else: status = 'processed_no_crop'
                frame_data_list.append({'frame': frame_count + 1, 'timestamp': timestamp, 'area': current_area, 'status': status})
                status_text = f"Frame: {frame_count+1} Area: {int(current_area_for_display)} State: {blink_state} (Total: {blink_count})"
            
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Max Area Ref: {int(max_observed_area)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            out_video.write(display_frame)
            progress_bar.progress((frame_count + 1) / total_frames, text=f"Analyzing... {frame_count+1}/{total_frames}")
            
        cap.release(); out_video.release(); progress_bar.empty()
        
        if not frame_data_list: return None, None, None, None
        df = pd.DataFrame(frame_data_list)
        analyzable_frames = df[df['status'] != 'blurry']
        if analyzable_frames.empty: return None, None, None, None

        positive_areas = analyzable_frames[analyzable_frames['area'] > 0]['area']
        true_max_area = positive_areas.quantile(0.995) if not positive_areas.empty else 1
        if pd.isna(true_max_area) or true_max_area == 0: true_max_area = 1
        
        df['normalized_area'] = df['area'].apply(lambda x: x / true_max_area if x >= 0 else -1)
        average_normalized_area = analyzable_frames['area'].clip(lower=0).mean() / true_max_area
        total_time_seconds = analyzable_frames['timestamp'].max() - analyzable_frames['timestamp'].min()
        blink_frequency_hz = blink_count / total_time_seconds if total_time_seconds > 0 else 0
        average_local_minimum_area = np.mean(local_minima_areas) if local_minima_areas else np.nan
        normalized_average_local_minimum = average_local_minimum_area / true_max_area if true_max_area > 0 and not pd.isna(average_local_minimum_area) else np.nan

        stats = {
            'total_blinks': blink_count, 'analysis_duration_s': total_time_seconds,
            'blink_frequency_per_min': blink_frequency_hz * 60,
            'average_normalized_area': average_normalized_area,
            'avg_local_min_normalized_area': normalized_average_local_minimum
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        
        plot_data = df[df['status'] == 'processed'] 
        if not plot_data.empty:
            ax.plot(plot_data['timestamp'], plot_data['normalized_area'], label='Normalized Eye Fissure Area')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalized Area (relative to max)'); ax.set_title('Normalized Eye Fissure Area Over Time')
            ax.set_ylim(bottom=-0.05, top=1.1); ax.legend(); ax.grid(True)
        
        with open(output_video_path, 'rb') as f: video_bytes = f.read()
        return video_bytes, df, fig, stats


# --- Streamlit UI 主函数 ---
def show_main_app():
    st.title("👁️ Mouse Blink Analysis Platform")
    st.markdown("Upload a video file, adjust the analysis thresholds if needed, and click 'Start Analysis'.")

    # 初始化 session_state，用于在 reruns 之间保存分析结果
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("1. Upload Video File")
        uploaded_file = st.file_uploader("Select a video file", type=["mp4", "avi", "mov"], label_visibility="collapsed")
        
        # 当上传新文件时，清空旧的分析结果
        if uploaded_file is not None:
             st.session_state.analysis_results = None

        st.subheader("2. General Thresholds")
        yolo_conf_threshold = st.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
        seg_threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.5, 0.01)
        laplacian_var_threshold = st.number_input("Blur Detection Threshold (Laplacian)", min_value=0, value=15)
        st.markdown("---")
        st.subheader("3. Blink Detection Logic")
        blink_drop_threshold = st.number_input("Blink Trigger (Area Drop)", min_value=50, value=200, step=10, help="Area decrease required to start a blink detection.")
        blink_confirm_ratio = st.slider("Blink Confirm (Closure Ratio)", 0.0, 1.0, 0.25, 0.05, help="Area must drop below this ratio of max area to be a confirmed blink.")
        reopen_ratio_threshold = st.slider("Blink Reset (Re-open Ratio)", 0.0, 1.0, 0.4, 0.05, help="Eye must re-open to this ratio of max area to reset the detector.")
        st.markdown("---")
        start_button = st.button("🚀 Start Analysis", disabled=(uploaded_file is None), type="primary")

    # --- 按钮点击后的处理逻辑 ---
    if start_button:
        init_db()
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                video_path = tmpfile.name
            st.info(f"Video file uploaded: {uploaded_file.name}. Processing...")
            
            with st.spinner("Loading models..."):
                yolo_model = load_detection_model(YOLO_MODEL_FILENAME)
                seg_model = load_segmentation_model(SEG_MODEL_FILENAME)
            
            if yolo_model and seg_model:
                config = {
                    'YOLO_CONF_THRESHOLD': yolo_conf_threshold, 'SEG_THRESHOLD': seg_threshold,
                    'LAPLACIAN_VAR_THRESHOLD': laplacian_var_threshold, 'BLINK_DROP_THRESHOLD': blink_drop_threshold,
                    'REOPEN_RATIO_THRESHOLD': reopen_ratio_threshold, 'BLINK_CONFIRM_RATIO': blink_confirm_ratio,
                    'MIN_OPEN_AREA_FOR_REF': 100,
                    'DIP_THRESHOLD': 50,
                }
                
                # 运行分析并将结果存入 session_state
                video_bytes, results_df, results_fig, stats = run_analysis(video_path, yolo_model, seg_model, config)
                os.remove(video_path)

                if video_bytes:
                    st.session_state.analysis_results = {
                        "video_bytes": video_bytes,
                        "results_df": results_df,
                        "results_fig": results_fig,
                        "stats": stats
                    }
                    try:
                        save_results_to_db(uploaded_file.name, stats)
                        st.success("Analysis summary saved to local history.")
                    except Exception as e:
                        st.warning(f"Could not save results to local history: {e}")
                else:
                    st.error("Analysis failed. The video might be corrupted or no eyes were detected.")
                    st.session_state.analysis_results = None # 失败时清空结果
        else:
            st.warning("Please upload a video file first.")

    # --- 结果展示区 ---
    # 独立于按钮，只要 session_state 中有结果，就显示出来
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        stats = results["stats"]
        video_bytes = results["video_bytes"]
        results_fig = results["results_fig"]
        results_df = results["results_df"]
        
        st.header("📊 Analysis Results")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Total Blinks", f"{stats['total_blinks']}")
        col2.metric("Blink Rate (per min)", f"{stats['blink_frequency_per_min']:.2f}")
        col3.metric("Avg Norm. Area", f"{stats['average_normalized_area']:.4f}", help="Average eye openness, normalized to max area.")
        col4.metric("Analyzed Duration", f"{stats['analysis_duration_s']:.2f} s")
        
        avg_dip_value = stats.get('avg_local_min_normalized_area')
        col5.metric(
            "Avg Dip Closure", 
            f"{avg_dip_value:.4f}" if avg_dip_value is not None and not np.isnan(avg_dip_value) else "N/A",
            help="Average depth of all minor/major eye closures (dips). Lower is deeper."
        )
        
        st.subheader("Processed Video"); st.video(video_bytes)
        st.download_button("📥 Download Processed Video", video_bytes, "processed_video.mp4", "video/mp4")
        
        st.subheader("Normalized Eye Fissure Area Over Time"); st.pyplot(results_fig)
        buf = BytesIO(); results_fig.savefig(buf, format="png")
        st.download_button("📥 Download Plot", buf, "eye_area_plot.png", "image/png")
        
        st.subheader("Detailed Frame-by-Frame Data"); st.dataframe(results_df, use_container_width=True)
        csv = results_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download CSV Data", csv, "blink_analysis_results.csv", "text/csv")

    # --- 本地历史记录 ---
    st.markdown("---")
    st.header("📜 Local Analysis History")
    init_db()
    history_df = load_results_from_db()
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No past analysis records found.")

# --- 脚本执行入口 ---
if __name__ == "__main__":
    show_main_app()
# -------------------- 请复制下面的整个函数 --------------------

def run_analysis(video_path, yolo_model, seg_model, config):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_video_path = os.path.join(temp_dir, 'processed_video.mp4')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): st.error("Error: Could not open the uploaded video file."); return None, None, None, None
        frame_width, frame_height, fps, total_frames = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        results_data = []
        max_observed_area = 0
        blink_count = 0
        previous_area = -1
        blink_state = 'OPEN'
        
        progress_bar = st.progress(0, text="Analyzing video...")
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            timestamp = (frame_count + 1) / fps; display_frame = frame.copy()
            
            # ✅ 核心修复：在循环开始时就为 status_text 提供一个默认值
            status_text = f"Frame: {frame_count+1} Initializing..."
            
            is_blurry = calculate_laplacian_variance(frame) < config['LAPLACIAN_VAR_THRESHOLD']
            
            in_blink_phase = (blink_state != 'OPEN')

            if is_blurry:
                results_data.append({'frame': frame_count + 1, 'timestamp': timestamp, 'area': -1, 'is_blinking': in_blink_phase, 'status': 'blurry'})
                status_text = f"Frame: {frame_count+1} Status: BLURRY (Total: {blink_count})"
            else:
                yolo_results = yolo_model.predict(frame, conf=config['YOLO_CONF_THRESHOLD'], classes=[0], verbose=False)
                eye_box = yolo_results[0].boxes[0].xyxy[0].cpu().numpy().astype(int) if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0 else None
                current_area = 0
                if eye_box is not None:
                    x1, y1, x2, y2 = eye_box; eye_crop = frame[y1:y2, x1:x2]
                    if eye_crop.size > 0:
                        input_tensor = preprocess_for_segmentation(eye_crop)
                        with torch.no_grad(): seg_output = seg_model(input_tensor)
                        binary_mask, current_area = postprocess_segmentation(seg_output, eye_crop.shape[:2], config['SEG_THRESHOLD'])
                        
                        if current_area > config['MIN_OPEN_AREA_FOR_REF']: max_observed_area = max(max_observed_area, current_area)
                        
                        if previous_area != -1 and max_observed_area > config['MIN_OPEN_AREA_FOR_REF']:
                            area_drop = previous_area - current_area
                            area_rise = current_area - previous_area 
                        
                            if blink_state == 'OPEN':
                                if area_drop > config['BLINK_DROP_THRESHOLD']: blink_state = 'CLOSING'
                            elif blink_state == 'CLOSING':
                                if max_observed_area > 0 and current_area < (max_observed_area * config['BLINK_CONFIRM_RATIO']): blink_state = 'CLOSED_CONFIRMED'
                                elif area_rise > 0: blink_state = 'OPEN'
                            elif blink_state == 'CLOSED_CONFIRMED':
                                if area_rise > 0:
                                    blink_count += 1
                                    blink_state = 'OPENING'
                            elif blink_state == 'OPENING':
                                if current_area > (max_observed_area * config['REOPEN_RATIO_THRESHOLD']): blink_state = 'OPEN'
                        
                        previous_area = current_area
                        in_blink_phase = (blink_state != 'OPEN')

                        if current_area > 0:
                            colored_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR); colored_mask[binary_mask == 255] = (255, 0, 255)
                            display_frame[y1:y2, x1:x2] = cv2.addWeighted(eye_crop, 0.7, colored_mask, 0.3, 0)
                        
                        results_data.append({'frame': frame_count + 1, 'timestamp': timestamp, 'area': current_area, 'is_blinking': in_blink_phase, 'status': 'processed'})
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else: 
                        results_data.append({'frame': frame_count + 1, 'timestamp': timestamp, 'area': -1, 'is_blinking': in_blink_phase, 'status': 'processed_no_crop'})
                else: 
                    results_data.append({'frame': frame_count + 1, 'timestamp': timestamp, 'area': -1, 'is_blinking': in_blink_phase, 'status': 'no_eye'})
                
                # 更新状态文本
                status_text = f"Frame: {frame_count+1} Area: {int(current_area)} State: {blink_state} (Total: {blink_count})"
            
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); cv2.putText(display_frame, f"Max Area Ref: {int(max_observed_area)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            out_video.write(display_frame); progress_bar.progress((frame_count + 1) / total_frames, text=f"Analyzing... {frame_count+1}/{total_frames}")
            
        cap.release(); out_video.release(); progress_bar.empty()
        
        # ... (函数剩余部分保持不变) ...
        df = pd.DataFrame(results_data)
        true_max_area = df[df['area'] > 0]['area'].max(); true_max_area = 1 if pd.isna(true_max_area) or true_max_area == 0 else true_max_area
        processed_frames_df = df[df['status'] == 'processed']; average_area = processed_frames_df['area'].mean() if not processed_frames_df.empty and processed_frames_df['area'].sum() > 0 else 0
        df.loc[:, 'normalized_area'] = df['area'].apply(lambda x: x / true_max_area if x >= 0 else -1)
        valid_frames = df[(df['status'] == 'processed') & (df['area'] >= 0)]; total_time_seconds = valid_frames['timestamp'].max() - valid_frames['timestamp'].min() if not valid_frames.empty else 0
        blink_frequency_hz = blink_count / total_time_seconds if total_time_seconds > 0 else 0
        stats = {'total_blinks': blink_count, 'analysis_duration': total_time_seconds, 'blink_frequency_hz': blink_frequency_hz, 'blink_frequency_per_min': blink_frequency_hz * 60, 'average_area': average_area}
        fig, ax = plt.subplots(figsize=(12, 6)); plot_data = df[df['normalized_area'] >= 0]
        if not plot_data.empty:
            ax.plot(plot_data['timestamp'], plot_data['normalized_area'], label='Normalized Eye Fissure Area')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalized Area (relative to max)'); ax.set_title('Normalized Eye Fissure Area Over Time'); ax.set_ylim(bottom=-0.05, top=1.1); ax.legend(); ax.grid(True)
        with open(output_video_path, 'rb') as f: video_bytes = f.read()
        return video_bytes, df, fig, stats
from flask import Flask, render_template, jsonify, request, send_file
import cv2
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import pyresearch 
app = Flask(__name__)

# Global analytics data (persistent across runs - in production, use DB)
analytics_data = {
    "accuracy": 80,
    "pests": 0,
    "fungus": 0,
    "scans": 0,
    "history": [],  # To store scan history
    "chart_labels": ["Pests", "Fungus", "False Positives", "Total Detections"],
    "chart_data": [0, 0, 0, 0]
}

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/api/analytics')
def get_analytics():
    return jsonify(analytics_data)

@app.route('/output_video')
def output_video():
    return send_file('static/output.mp4', mimetype='video/mp4', conditional=True)

@app.route('/run_detection', methods=['GET', 'POST'])
def run_detection():
    try:
        # Load YOLO model (two classes: Pest and Fungus)
        model = YOLO('best.pt')
        
        # Allow video upload or use demo
        if request.method == 'POST' and 'video' in request.files:
            video_file = request.files['video']
            video_path = 'static/uploaded_video.mp4'
            video_file.save(video_path)
        else:
            video_path = 'static/demo.mp4'
        
        output_path = 'static/output.mp4'
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Video not found or upload failed.'})
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Use 'avc1' for better browser compatibility (H.264 codec)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        pest_count = 0
        fungus_count = 0
        false_positives = 0  # No other classes, so this should be 0
        
        frame_count = 0
        severity_levels = []  # Estimate severity based on density
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            detections_in_frame = 0
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = model.names[cls_id].lower()
                    detections_in_frame += 1
                    
                    if 'pest' in cls_name:
                        pest_count += 1
                    elif 'fungus' in cls_name:
                        fungus_count += 1
                    else:
                        false_positives += 1  # Should be 0 for two-class model
            
            # Simple severity estimation (density-based)
            severity = min(100, (detections_in_frame / (frame_width * frame_height / 100000)) * 100)
            severity_levels.append(severity)
            
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        
        total_detections = pest_count + fungus_count + false_positives
        avg_severity = np.mean(severity_levels) if severity_levels else 0
        
        # Update analytics
        analytics_data['pests'] += pest_count
        analytics_data['fungus'] += fungus_count
        analytics_data['scans'] += 1
        analytics_data['chart_data'] = [analytics_data['pests'], analytics_data['fungus'], false_positives, total_detections]
        
        # Add to history
        scan_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pests": pest_count,
            "fungus": fungus_count,
            "severity": round(avg_severity, 2)
        }
        analytics_data['history'].append(scan_entry)
        if len(analytics_data['history']) > 10:
            analytics_data['history'] = analytics_data['history'][-10:]
        
        return jsonify({
            'success': True,
            'pests': pest_count,
            'fungus': fungus_count,
            'severity': round(avg_severity, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
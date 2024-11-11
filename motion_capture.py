import cv2
import mediapipe as mp
import json
import tkinter as tk
from tkinter import filedialog, ttk
import os
from collections import deque

class MotionCaptureApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Motion Capture Interface")
        self.root.geometry("600x400")
        
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.smoothing_factor = tk.DoubleVar(value=0.5)
        self.min_detection_confidence = tk.DoubleVar(value=0.5)
        self.buffer_size = 5
        
        self.setup_ui()
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.keypoint_buffers = {}
        
    def setup_ui(self):
        input_frame = ttk.LabelFrame(self.root, text="Input Video", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        ttk.Entry(input_frame, textvariable=self.input_path).pack(side="left", fill="x", expand=True)
        ttk.Button(input_frame, text="Browse", command=self.select_input).pack(side="right")
        
        output_frame = ttk.LabelFrame(self.root, text="Output JSON", padding="10")
        output_frame.pack(fill="x", padx=10, pady=5)
        ttk.Entry(output_frame, textvariable=self.output_path).pack(side="left", fill="x", expand=True)
        ttk.Button(output_frame, text="Browse", command=self.select_output).pack(side="right")
        
        params_frame = ttk.LabelFrame(self.root, text="Parameters", padding="10")
        params_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(params_frame, text="Smoothing Factor:").pack()
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.smoothing_factor, orient="horizontal").pack(fill="x")
        ttk.Label(params_frame, text="Detection Confidence:").pack()
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.min_detection_confidence, orient="horizontal").pack(fill="x")
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x", padx=10, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var).pack()
        ttk.Button(self.root, text="Process Video", command=self.process_video).pack(pady=10)
        
    def select_input(self):
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")])
        if filename:
            self.input_path.set(filename)
            default_output = os.path.splitext(filename)[0] + "_motion_data.json"
            self.output_path.set(default_output)
            
    def select_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if filename:
            self.output_path.set(filename)
            
    def smooth_keypoints(self, new_keypoints):
        smoothed_keypoints = {}
        for point_id in new_keypoints:
            if point_id not in self.keypoint_buffers:
                self.keypoint_buffers[point_id] = deque(maxlen=self.buffer_size)
            self.keypoint_buffers[point_id].append(new_keypoints[point_id])
            if len(self.keypoint_buffers[point_id]) > 0:
                smoothed_point = {'x': 0, 'y': 0, 'z': 0, 'visibility': 0}
                total_weight = 0
                for i, point in enumerate(self.keypoint_buffers[point_id]):
                    weight = (i + 1) / len(self.keypoint_buffers[point_id])
                    total_weight += weight
                    for key in smoothed_point:
                        smoothed_point[key] += point[key] * weight
                for key in smoothed_point:
                    smoothed_point[key] /= total_weight
                smoothed_keypoints[point_id] = smoothed_point
        return smoothed_keypoints
    
    def process_video(self):
        if not self.input_path.get() or not self.output_path.get():
            self.status_var.set("Please select input and output files")
            return
            
        self.status_var.set("Processing...")
        self.progress_var.set(0)
        self.root.update()
        
        # Initialize pose detection
        pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.min_detection_confidence.get(),
            min_tracking_confidence=self.min_detection_confidence.get()
        )
        
        # Open video
        cap = cv2.VideoCapture(self.input_path.get())
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keypoints_data = []
        
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            progress = (frame_index / total_frames) * 100
            self.progress_var.set(progress)
            self.root.update()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            
            if result.pose_landmarks:
                frame_keypoints = {}
                for i, landmark in enumerate(result.pose_landmarks.landmark):
                    frame_keypoints[f'point_{i}'] = {'x': landmark.x, 'y': landmark.y, 'z': landmark.z, 'visibility': landmark.visibility}
                
                smoothed_keypoints = self.smooth_keypoints(frame_keypoints)
                keypoints_data.append({'frame_index': frame_index, 'keypoints': smoothed_keypoints})
                
                self.mp_drawing.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
            cv2.imshow("Processing Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_index += 1
            
        cap.release()
        cv2.destroyAllWindows()
        
        with open(self.output_path.get(), 'w') as f:
            json.dump(keypoints_data, f, indent=4)
            
        self.status_var.set(f"Processing complete. Saved to {self.output_path.get()}")
        self.progress_var.set(100)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MotionCaptureApp()
    app.run()

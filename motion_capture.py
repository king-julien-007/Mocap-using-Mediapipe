import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, ttk
import os
from collections import deque
from datetime import datetime

class MotionCaptureApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Motion Capture Interface")
        self.root.geometry("700x500")
        self.root.configure(bg="#1e1e1e")
        
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.smoothing_factor = tk.DoubleVar(value=0.5)
        self.min_detection_confidence = tk.DoubleVar(value=0.5)
        self.buffer_size = 5
        self.cancel_processing = False  # Flag to control the cancellation

        self.setup_ui()

        self.mp_pose = mp.solutions.pose
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS
        self.keypoint_buffers = {}

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="white", font=("Montserrat", 10))
        style.configure("TButton", font=("Montserrat", 10), background="#444", foreground="white")
        style.configure("TEntry", fieldbackground="#444", foreground="white")
        style.configure("TScale", background="#1e1e1e")
        style.configure("TLabelFrame", background="#1e1e1e", foreground="white", font=("Montserrat", 10))

        input_frame = ttk.LabelFrame(self.root, text="Input Video", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        ttk.Entry(input_frame, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(input_frame, text="Browse", command=self.select_input).pack(side="right", padx=5)

        output_frame = ttk.LabelFrame(self.root, text="Output File", padding="10")
        output_frame.pack(fill="x", padx=10, pady=5)
        ttk.Entry(output_frame, textvariable=self.output_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(output_frame, text="Browse", command=self.select_output).pack(side="right", padx=5)

        params_frame = ttk.LabelFrame(self.root, text="Parameters", padding="10")
        params_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(params_frame, text="Smoothing Factor:").pack()
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.smoothing_factor, orient="horizontal").pack(fill="x")
        ttk.Label(params_frame, text="Detection Confidence:").pack()
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.min_detection_confidence, orient="horizontal").pack(fill="x")

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100, length=300)
        self.progress.pack(fill="x", padx=10, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, font=("Montserrat", 10)).pack(pady=5)
        ttk.Button(self.root, text="Process Video", command=self.process_video).pack(pady=10)
        ttk.Button(self.root, text="Cancel", command=self.cancel_process).pack(pady=10)  # Add cancel button

    def cancel_process(self):
        self.cancel_processing = True

    def select_input(self):
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")])
        if filename:
            self.input_path.set(filename)
            default_output = os.path.splitext(filename)[0] + "_motion_data.bvh"
            self.output_path.set(default_output)

    def select_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".bvh", filetypes=[("BVH files", "*.bvh"), ("All files", "*.*")])
        if filename:
            self.output_path.set(filename)

    def smooth_keypoints(self, new_keypoints):
        smoothed_keypoints = {}
        for point_id in new_keypoints:
            if point_id not in self.keypoint_buffers:
                self.keypoint_buffers[point_id] = deque(maxlen=self.buffer_size)
            self.keypoint_buffers[point_id].append(new_keypoints[point_id])
            smoothed_point = {key: 0 for key in new_keypoints[point_id]}
            for i, point in enumerate(self.keypoint_buffers[point_id]):
                weight = (i + 1) / len(self.keypoint_buffers[point_id])
                for key in smoothed_point:
                    smoothed_point[key] += point[key] * weight
            smoothed_keypoints[point_id] = smoothed_point
        return smoothed_keypoints

    def process_video(self):
        if not self.input_path.get() or not self.output_path.get():
            self.status_var.set("Please select input and output files")
            return

        self.status_var.set("Processing...")
        self.progress_var.set(0)
        self.root.update()
        self.cancel_processing = False  # Reset the cancel flag

        pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.min_detection_confidence.get(),
            min_tracking_confidence=self.min_detection_confidence.get()
        )

        cap = cv2.VideoCapture(self.input_path.get())
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keypoints_data = []

        frame_index = 0
        while cap.isOpened():
            if self.cancel_processing:
                self.status_var.set("Processing cancelled")
                break

            ret, frame = cap.read()
            if not ret:
                break

            progress = (frame_index / total_frames) * 100
            self.progress_var.set(progress)
            self.root.update()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            if result.pose_landmarks:
                frame_keypoints = {f'point_{i}': {'x': lm.x, 'y': lm.y, 'z': lm.z} for i, lm in enumerate(result.pose_landmarks.landmark)}
                smoothed_keypoints = self.smooth_keypoints(frame_keypoints)
                keypoints_data.append(smoothed_keypoints)

                for connection in self.pose_connections:
                    pt1 = result.pose_landmarks.landmark[connection[0]]
                    pt2 = result.pose_landmarks.landmark[connection[1]]
                    cv2.line(frame, (int(pt1.x * frame.shape[1]), int(pt1.y * frame.shape[0])),
                             (int(pt2.x * frame.shape[1]), int(pt2.y * frame.shape[0])), (169, 169, 169), 1, cv2.LINE_AA)

                for landmark in result.pose_landmarks.landmark:
                    cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

                # Display detection confidence level
                confidence_text = f"Detection Confidence: {result.pose_landmarks.landmark[0].visibility:.2f}"
                cv2.putText(frame, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Processing Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

        if not self.cancel_processing:
            self.export_to_bvh(keypoints_data)
            self.status_var.set(f"Processing complete. Saved to {self.output_path.get()}")
            self.progress_var.set(100)

    def export_to_bvh(self, keypoints_data):
        with open(self.output_path.get(), 'w') as f:
            f.write("HIERARCHY\n")
            f.write("ROOT Hips\n")
            f.write("{\n")
            f.write("\tOFFSET 0.00 0.00 0.00\n")
            f.write("\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
            for i in range(1, 33):  # Assuming 33 joints
                f.write(f"\tJOINT Joint{i}\n")
                f.write("\t{\n")
                f.write("\t\tOFFSET 0.00 0.00 0.00\n")
                f.write("\t\tCHANNELS 3 Zrotation Xrotation Yrotation\n")
                f.write("\t\tEnd Site\n")
                f.write("\t\t{\n")
                f.write("\t\t\tOFFSET 0.00 0.00 0.00\n")
                f.write("\t\t}\n")
                f.write("\t}\n")
            f.write("}\n")
            f.write("MOTION\n")
            f.write(f"Frames: {len(keypoints_data)}\n")
            f.write("Frame Time: 0.0333333\n")  # Assuming 30 FPS

            for frame in keypoints_data:
                frame_data = []
                for point_id in frame:
                    point = frame[point_id]
                    frame_data.extend([point['x'] * 100, point['y'] * 100, point['z'] * 100])  # Scale positions
                    frame_data.extend([0.0, 0.0, 0.0])  # Placeholder for rotations
                f.write(" ".join(map(str, frame_data)) + "\n")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MotionCaptureApp()
    app.run()

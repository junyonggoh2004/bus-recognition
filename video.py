import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import queue
import tempfile
import os
import re

# Import your custom classes
from trocr import TrOCRInference
from similarity_search.similarity_search import SimilaritySearch


class VideoDetectionProcessor:
    def __init__(self, model_path="best_v6.pt"):
        # Initialize all attributes first
        self.model = YOLO(model_path)
        self.cap = None
        self.is_processing = False
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        # Detection settings
        self.confidence_threshold = 0.5
        self.track_history = defaultdict(lambda: [])

        # Frame processing
        self.frame_skip = 2  # Process every nth frame
        self.frame_count = 0

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Video display
        self.display_frame = None
        self.frame_lock = threading.Lock()

        # OCR and similarity search
        self.ocr_model = None
        self.similarity_search = SimilaritySearch()
        self.ocr_frame_skip = 30  # Process OCR every 30th frame
        self.last_ocr_result = ""

    def initialize_ocr(self, model_path="microsoft/trocr-base-printed"):
        """Initialize OCR model"""
        try:
            self.ocr_model = TrOCRInference(model_path)
            print("OCR model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading OCR model: {e}")
            return False

    def load_video(self, video_path):
        """Load video file"""
        try:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise Exception("Could not open video file")

            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0

            # Get frame dimensions
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(
                f"Video loaded: {self.total_frames} frames, {self.fps:.2f} FPS, {self.duration:.2f}s")
            print(f"Resolution: {self.frame_width}x{self.frame_height}")
            return True

        except Exception as e:
            print(f"Error loading video: {e}")
            return False

    def start_video_processing(self, progress_callback=None, result_callback=None, frame_callback=None, ocr_callback=None):
        """Start video processing in a separate thread"""
        if self.is_processing:
            return False

        if not self.cap or not self.cap.isOpened():
            return False

        self.is_processing = True
        self.stop_event.clear()

        self.processing_thread = threading.Thread(
            target=self._process_video_frames,
            args=(progress_callback, result_callback,
                  frame_callback, ocr_callback),
            daemon=True
        )
        self.processing_thread.start()
        return True

    def stop_video_processing(self):
        """Stop video processing"""
        if not hasattr(self, 'is_processing') or not self.is_processing:
            return

        self.is_processing = False
        if hasattr(self, 'stop_event'):
            self.stop_event.set()

        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

    def _process_video_frames(self, progress_callback=None, result_callback=None, frame_callback=None, ocr_callback=None):
        """Process video frames with object detection and OCR"""
        try:
            frame_count = 0
            detection_results = []
            last_time = time.time()

            # Reset video to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while self.is_processing and not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = time.time()

                # Process frame for detection
                processed_frame = frame.copy()

                # Resize frame for faster processing if needed
                height, width = frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    processed_frame = cv2.resize(
                        processed_frame, (new_width, new_height))

                # Run detection
                results = self.model.track(
                    processed_frame,
                    conf=self.confidence_threshold,
                    persist=True,
                    verbose=False
                )

                # Draw detections on original frame
                display_frame = self._draw_detections(
                    frame, results[0], scale_factor=width/processed_frame.shape[1] if width > 1280 else 1.0)

                # Store frame for display
                with self.frame_lock:
                    self.display_frame = display_frame.copy()

                # Send frame to GUI for display
                if frame_callback:
                    frame_callback(display_frame)

                # Process OCR every nth frame
                if self.ocr_model and frame_count % self.ocr_frame_skip == 0 and ocr_callback:
                    threading.Thread(target=self._process_ocr, args=(
                        frame.copy(), ocr_callback), daemon=True).start()

                # Process results for logging
                if frame_count % self.frame_skip == 0:
                    frame_results = self._process_detection_results(
                        results[0], frame_count)
                    if frame_results:
                        detection_results.extend(frame_results)

                # Update progress
                if progress_callback:
                    progress = (frame_count / self.total_frames) * 100
                    progress_callback(progress, frame_count, self.total_frames)

                # Send results periodically
                if result_callback and len(detection_results) >= 10:
                    result_callback(detection_results.copy())
                    detection_results.clear()

                # Control playback speed (approximate original FPS)
                # Control playback speed (slower than original FPS)
                if self.fps > 0:
                    # Slow down by factor (2 = half speed, 4 = quarter speed, etc.)
                    slowdown_factor = 4  # Adjust this value to control speed
                    target_time = (1.0 / self.fps) * slowdown_factor
                    elapsed = time.time() - current_time
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)

                # FPS calculation
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()

            # Send final results
            if result_callback and detection_results:
                result_callback(detection_results)

        except Exception as e:
            print(f"Error in video processing: {e}")
        finally:
            self.is_processing = False

    def _process_ocr(self, frame, ocr_callback):
        """Process frame with OCR in background thread"""
        try:
            # Save frame temporarily
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(
                temp_dir, f"temp_frame_{int(time.time() * 1000)}.jpg")

            cv2.imwrite(temp_path, frame)

            # Perform OCR
            ocr_result = self.ocr_model.inference(temp_path)

            # Clean up temp file
            os.remove(temp_path)

            # Skip if same as last result
            if ocr_result and ocr_result != self.last_ocr_result:
                self.last_ocr_result = ocr_result

                # Extract bus number from OCR text
                bus_number = self._extract_bus_number(ocr_result)

                if bus_number:
                    # Get route information and perform similarity search
                    route_info = self.similarity_search.extract_wiki_origin_and_destination(
                        bus_number)

                    if route_info and len(route_info) >= 2 and route_info[0] != 'Route not found':
                        # Perform fuzzy matching
                        locations = route_info
                        if len(locations) > 0:
                            matched_bus_number, best_match = self.similarity_search.match_ocr_text(
                                ocr_result, locations)

                            # Send result to callback
                            if ocr_callback:
                                ocr_callback(bus_number, best_match)

        except Exception as e:
            print(f"OCR processing error: {e}")

    def _extract_bus_number(self, text):
        """Extract bus number from OCR text"""
        # Common patterns for bus numbers
        patterns = [
            r'\b\d{1,4}[A-Z]?\b',  # 123, 123A, etc.
            r'\b[A-Z]\d{1,4}\b',   # A123, etc.
            r'\b\d{1,4}\b'         # Just numbers
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            if matches:
                # Return the first reasonable match
                for match in matches:
                    if len(match) >= 1 and len(match) <= 5:
                        return match

        return None

    def _process_detection_results(self, results, frame_number):
        """Process detection results for a single frame"""
        detections = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            # Get track IDs if available
            track_ids = None
            if hasattr(results.boxes, 'id') and results.boxes.id is not None:
                track_ids = results.boxes.id.cpu().numpy()

            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box
                    class_name = self.model.names[int(cls)]

                    track_id = int(
                        track_ids[i]) if track_ids is not None else None

                    detection = {
                        'frame': frame_number,
                        'timestamp': frame_number / self.fps if self.fps > 0 else 0,
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'track_id': track_id
                    }
                    detections.append(detection)

        return detections

    def get_current_display_frame(self):
        """Get current frame for display"""
        with self.frame_lock:
            return self.display_frame.copy() if self.display_frame is not None else None

    def _draw_detections(self, frame, results, scale_factor=1.0):
        """Draw detection boxes and labels on frame"""
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            # Get track IDs if available
            track_ids = None
            if hasattr(results.boxes, 'id') and results.boxes.id is not None:
                track_ids = results.boxes.id.cpu().numpy()

            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                if conf >= self.confidence_threshold:
                    # Scale coordinates back to original frame size if needed
                    x1, y1, x2, y2 = box * scale_factor
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    class_name = self.model.names[int(cls)]
                    track_id = int(
                        track_ids[i]) if track_ids is not None else None

                    # Choose color based on class
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
                              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    color = colors[int(cls) % len(colors)]

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Create label
                    label = f"{class_name}: {conf:.2f}"
                    if track_id is not None:
                        label += f" ID:{track_id}"

                    # Draw label background
                    label_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                  (x1 + label_size[0], y1), color, -1)

                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_video_processing()
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
        except:
            pass


class VideoDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Object Detection with Display")
        self.root.geometry("1400x900")

        self.processor = VideoDetectionProcessor()
        self.current_video_path = None
        self.detection_results = []

        # Video display variables
        self.video_label = None
        self.display_after_id = None

        self.setup_gui()
        self.initialize_ocr()

    def initialize_ocr(self):
        """Initialize OCR model in background"""
        def load_ocr():
            success = self.processor.initialize_ocr()
            if success:
                self.root.after(0, lambda: self.status_label.config(
                    text="Ready (OCR enabled)"))
            else:
                self.root.after(0, lambda: self.status_label.config(
                    text="Ready (OCR disabled)"))

        threading.Thread(target=load_ocr, daemon=True).start()

    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Control panel
        control_frame = ttk.LabelFrame(
            main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2,
                           sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)

        # File selection
        ttk.Button(control_frame, text="Select Video", command=self.select_video).grid(
            row=0, column=0, padx=(0, 5))
        self.video_path_label = ttk.Label(
            control_frame, text="No video selected")
        self.video_path_label.grid(row=0, column=1, padx=(5, 0), sticky=tk.W)

        # Processing controls
        self.start_button = ttk.Button(control_frame, text="Start Processing",
                                       command=self.start_processing, state=tk.DISABLED)
        self.start_button.grid(row=1, column=0, padx=(0, 5), pady=(5, 0))

        self.stop_button = ttk.Button(control_frame, text="Stop Processing",
                                      command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1, padx=(
            5, 0), pady=(5, 0), sticky=tk.W)

        # Settings and status frame
        middle_frame = ttk.Frame(main_frame)
        middle_frame.grid(row=1, column=0, columnspan=2, sticky=(
            tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        middle_frame.columnconfigure(1, weight=1)
        middle_frame.rowconfigure(0, weight=1)

        # Settings panel
        settings_frame = ttk.LabelFrame(
            middle_frame, text="Settings", padding="5")
        settings_frame.grid(row=0, column=0, sticky=(
            tk.W, tk.E, tk.N), padx=(0, 10))

        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence:").grid(
            row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=0.9,
                                     variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        self.confidence_label = ttk.Label(settings_frame, text="0.50")
        self.confidence_label.grid(row=0, column=2)
        self.confidence_var.trace('w', self.update_confidence)

        # Status panel
        status_frame = ttk.LabelFrame(middle_frame, text="Status", padding="5")
        status_frame.grid(row=0, column=1, sticky=(
            tk.W, tk.E, tk.N), padx=(10, 0))
        status_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(
            row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.status_label = ttk.Label(status_frame, text="Loading OCR...")
        self.status_label.grid(row=1, column=0, sticky=tk.W)

        self.frame_label = ttk.Label(status_frame, text="Frame: 0/0")
        self.frame_label.grid(row=2, column=0, sticky=tk.W)

        # Video display and results frame
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, columnspan=2,
                           sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)

        # Video display frame
        video_frame = ttk.LabelFrame(
            content_frame, text="Video Display", padding="5")
        video_frame.grid(row=0, column=0, sticky=(
            tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        # Video display canvas
        self.video_canvas = tk.Canvas(
            video_frame, bg='black', width=640, height=480)
        self.video_canvas.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Results frame
        results_frame = ttk.LabelFrame(
            content_frame, text="Detection Results", padding="5")
        results_frame.grid(row=0, column=1, sticky=(
            tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Results text with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.results_text = tk.Text(text_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Export button
        ttk.Button(results_frame, text="Export Results",
                   command=self.export_results).grid(row=1, column=0, pady=(10, 0))

    def select_video(self):
        """Select video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_video_path = file_path
            if self.processor.load_video(file_path):
                filename = file_path.split('/')[-1]
                self.video_path_label.config(text=f"Video: {filename}")
                self.start_button.config(state=tk.NORMAL)
                ocr_status = "OCR enabled" if self.processor.ocr_model else "OCR disabled"
                self.status_label.config(text=f"Video loaded ({ocr_status})")

                # Clear video display
                self.video_canvas.delete("all")
                self.video_canvas.create_text(
                    self.video_canvas.winfo_width()//2,
                    self.video_canvas.winfo_height()//2,
                    text="Video loaded. Click 'Start Processing' to begin.",
                    fill="white", font=("Arial", 12)
                )
            else:
                messagebox.showerror("Error", "Failed to load video file")

    def start_processing(self):
        """Start video processing"""
        if not self.current_video_path:
            return

        # Update confidence threshold
        self.processor.confidence_threshold = self.confidence_var.get()

        # Clear previous results
        self.detection_results.clear()
        self.results_text.delete(1.0, tk.END)

        # Start processing
        success = self.processor.start_video_processing(
            progress_callback=self.update_progress,
            result_callback=self.update_results,
            frame_callback=self.update_video_display,
            ocr_callback=self.update_ocr_results
        )

        if success:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Processing video...")
        else:
            messagebox.showerror("Error", "Failed to start video processing")

    def stop_processing(self):
        """Stop video processing"""
        self.processor.stop_video_processing()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        ocr_status = "OCR enabled" if self.processor.ocr_model else "OCR disabled"
        self.status_label.config(text=f"Processing stopped ({ocr_status})")

        # Cancel any pending display updates
        if self.display_after_id:
            self.root.after_cancel(self.display_after_id)

    def update_video_display(self, frame):
        """Update video display with new frame"""
        if frame is not None:
            # Convert frame to display format
            self.root.after_idle(self._display_frame, frame)

    def _display_frame(self, frame):
        """Display frame on canvas (called from main thread)"""
        try:
            # Get canvas dimensions
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not initialized yet
                self.root.after(100, lambda: self._display_frame(frame))
                return

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame to fit canvas while maintaining aspect ratio
            frame_height, frame_width = frame_rgb.shape[:2]

            # Calculate scaling factor
            scale_w = canvas_width / frame_width
            scale_h = canvas_height / frame_height
            scale = min(scale_w, scale_h)

            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)

            # Resize frame
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

            # Convert to PhotoImage
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)

            # Clear canvas and display image
            self.video_canvas.delete("all")

            # Center the image
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2

            self.video_canvas.create_image(x, y, image=photo, anchor=tk.NW)

            # Keep a reference to prevent garbage collection
            self.video_canvas.image = photo

        except Exception as e:
            print(f"Error displaying frame: {e}")

    def update_progress(self, progress, current_frame, total_frames):
        """Update progress bar and status"""
        self.progress_var.set(progress)
        self.frame_label.config(text=f"Frame: {current_frame}/{total_frames}")

        if progress >= 100:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            ocr_status = "OCR enabled" if self.processor.ocr_model else "OCR disabled"
            self.status_label.config(
                text=f"Processing completed ({ocr_status})")

    def update_results(self, new_results):
        """Update results display"""
        self.detection_results.extend(new_results)

        # Update text display
        for result in new_results[-5:]:  # Show last 5 detections
            timestamp = result['timestamp']
            class_name = result['class']
            confidence = result['confidence']
            track_id = result.get('track_id', 'N/A')

            text = f"[{timestamp:.2f}s] {class_name} (conf: {confidence:.2f}, ID: {track_id})\n"
            self.results_text.insert(tk.END, text)

        self.results_text.see(tk.END)

    def update_ocr_results(self, bus_number, destination):
        """Update OCR results display - only show bus number and destination"""
        timestamp = time.strftime("%H:%M:%S")
        text = f"[{timestamp}] Bus {bus_number} -> {destination}\n"
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)

    def update_confidence(self, *args):
        """Update confidence threshold display"""
        value = self.confidence_var.get()
        self.confidence_label.config(text=f"{value:.2f}")

    def export_results(self):
        """Export detection results to file"""
        if not self.detection_results:
            messagebox.showwarning("Warning", "No results to export")
            return

        file_path = filedialog.asksavename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    if file_path.endswith('.csv'):
                        f.write(
                            "timestamp,frame,class,confidence,x1,y1,x2,y2,track_id\n")
                        for result in self.detection_results:
                            bbox = result['bbox']
                            f.write(f"{result['timestamp']:.2f},{result['frame']},"
                                    f"{result['class']},{result['confidence']:.2f},"
                                    f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f},"
                                    f"{result.get('track_id', '')}\n")
                    else:
                        for result in self.detection_results:
                            f.write(f"Frame {result['frame']} ({result['timestamp']:.2f}s): "
                                    f"{result['class']} (confidence: {result['confidence']:.2f})\n")

                messagebox.showinfo(
                    "Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {e}")

    def on_closing(self):
        """Handle window closing"""
        if self.processor.is_processing:
            self.stop_processing()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = VideoDetectionGUI(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
    finally:
        # Cleanup
        if hasattr(app, 'processor'):
            app.processor.stop_video_processing()


if __name__ == "__main__":
    main()

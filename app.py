import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from ultralytics import YOLO
import subprocess
import easyocr
from similarity_search.similarity_search import SimilaritySearch
from tts.tts import TTS
from trocr import TrOCRInference


class BusDetectionApp:
    def __init__(self, root):

        self.ss = SimilaritySearch()
        self.tts = TTS()
        self.trocr = TrOCRInference("trocr-finetuned")

        self.root = root
        self.root.title("Bus Detection App")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        # Variables
        self.current_image = None
        self.original_image = None
        self.image_path = None
        self.current_page = "main"  # Track current page

        # Load YOLO model
        self.load_yolo_model()

        # Create main menu
        self.create_main_menu()

        output_dir = "tk_cropped_outputs"
        os.makedirs(output_dir, exist_ok=True)

        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    def load_yolo_model(self):
        """Load YOLO model for object detection"""
        try:
            # Load your trained YOLO model
            model_path = "best_v6.pt"  # Make sure this file is in the same directory

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file '{model_path}' not found. Please make sure best.pt is in the same directory as this script.")

            self.model = YOLO(model_path)
            print(f"YOLO model loaded successfully from {model_path}")

            # Get class names from the model
            self.classes = list(self.model.names.values()) if hasattr(
                self.model, 'names') else ['bus']
            print(f"Model classes: {self.classes}")

        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            messagebox.showerror(
                "Model Error", f"Failed to load YOLO model: {str(e)}\n\nPlease make sure:\n1. best.pt file is in the same directory\n2. ultralytics library is installed: pip install ultralytics")
            self.model = None

    def create_main_menu(self):
        """Create the main menu page"""
        self.clear_page()
        self.current_page = "main"

        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both')

        # Title
        title_label = tk.Label(
            main_frame,
            text="Bus Detection App",
            font=("Arial", 32, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=50)

        # Subtitle
        # subtitle_label = tk.Label(
        #     main_frame,
        #     text="Choose your detection mode",
        #     font=("Arial", 16),
        #     bg='#f0f0f0',
        #     fg='#666666'
        # )
        # subtitle_label.pack(pady=10)

        # Buttons container
        buttons_frame = tk.Frame(main_frame, bg='#f0f0f0')
        buttons_frame.pack(pady=50)

        # Image detection button
        image_btn = tk.Button(
            buttons_frame,
            text="📸 Image Detection",
            command=self.open_image_detection,
            font=("Arial", 18, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=40,
            pady=20,
            cursor='hand2',
            relief='raised',
            bd=3
        )
        image_btn.pack(pady=15)

        # Video detection button
        # video_btn = tk.Button(
        #     buttons_frame,
        #     text="🎥 Video Detection",
        #     command=self.open_video_detection,
        #     font=("Arial", 18, "bold"),
        #     bg='#2196F3',
        #     fg='white',
        #     padx=40,
        #     pady=20,
        #     cursor='hand2',
        #     relief='raised',
        #     bd=3
        # )
        # video_btn.pack(pady=15)

        # Info label
        # info_label = tk.Label(
        #     main_frame,
        #     text="Select Image Detection to detect bus in static images\nSelect Video Detection to detect bus in video files",
        #     font=("Arial", 12),
        #     bg='#f0f0f0',
        #     fg='#888888',
        #     justify='center'
        # )
        # info_label.pack(pady=30)

    def open_image_detection(self):
        """Open image detection page"""
        self.create_image_detection_page()

    def open_video_detection(self):
        """Open video detection page"""
        self.create_video_detection_page()

    def clear_page(self):
        """Clear all widgets from the current page"""
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_image_detection_page(self):
        """Create the image detection interface"""
        self.clear_page()
        self.current_page = "image"

        # Back button
        back_btn = tk.Button(
            self.root,
            text="← Back to Main Menu",
            command=self.create_main_menu,
            font=("Arial", 12),
            bg='#f44336',
            fg='white',
            padx=15,
            pady=5,
            cursor='hand2'
        )
        back_btn.pack(anchor='nw', padx=10, pady=10)

        # Upload section
        upload_frame = tk.Frame(self.root, bg='#f0f0f0')
        upload_frame.pack(pady=20)

        upload_btn = tk.Button(
            upload_frame,
            text="Upload Image",
            command=self.upload_image,
            font=("Arial", 14),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        upload_btn.pack(side=tk.LEFT, padx=10)

        detect_btn = tk.Button(
            upload_frame,
            text="Detect Bus",
            command=self.detect_buses,
            font=("Arial", 14),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            state='disabled'
        )
        detect_btn.pack(side=tk.LEFT, padx=10)
        self.detect_btn = detect_btn

        # Image display area
        self.image_frame = tk.Frame(
            self.root, bg='white', relief='sunken', bd=2)
        self.image_frame.pack(pady=20, padx=20, fill='both', expand=True)

        self.image_label = tk.Label(
            self.image_frame,
            text="No image uploaded\nClick 'Upload Image' to get started",
            font=("Arial", 16),
            bg='white',
            fg='gray'
        )
        self.image_label.pack(expand=True)

        # Results area
        self.results_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.results_frame.pack(pady=10, padx=20, fill='x')

        # Create the results label before packing it
        self.results_label = tk.Label(
            self.results_frame,
            text="",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#333333'
        )
        self.results_label.pack()

        # Clear button
        # clear_btn = tk.Button(
        #     self.root,
        #     text="Clear Image",
        #     command=self.clear_image,
        #     font=("Arial", 12),
        #     bg='#FF9800',
        #     fg='white',
        #     padx=15,
        #     pady=5,
        #     cursor='hand2'
        # )
        # clear_btn.pack(pady=10)

    def create_video_detection_page(self):
        """Create the video detection interface"""
        self.clear_page()
        self.current_page = "video"

        # Back button
        back_btn = tk.Button(
            self.root,
            text="← Back to Main Menu",
            command=self.create_main_menu,
            font=("Arial", 12),
            bg='#f44336',
            fg='white',
            padx=15,
            pady=5,
            cursor='hand2'
        )
        back_btn.pack(anchor='nw', padx=10, pady=10)

        # Title
        title_label = tk.Label(
            self.root,
            text="Video Bus Detection",
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=50)

        # Coming soon message
        coming_soon_frame = tk.Frame(self.root, bg='#f0f0f0')
        coming_soon_frame.pack(expand=True)

        icon_label = tk.Label(
            coming_soon_frame,
            text="🚧",
            font=("Arial", 48),
            bg='#f0f0f0'
        )
        icon_label.pack(pady=20)

        message_label = tk.Label(
            coming_soon_frame,
            text="Video Detection Coming Soon!",
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        message_label.pack(pady=10)

        desc_label = tk.Label(
            coming_soon_frame,
            text="This feature is currently under development.\nPlease use Image Detection for now.",
            font=("Arial", 14),
            bg='#f0f0f0',
            fg='#666666',
            justify='center'
        )
        desc_label.pack(pady=20)

        # Placeholder for video detection code
        self.process_video_detection()

    def process_video_detection(self):
        """Process video detection - placeholder for future implementation"""
        # Video detection code will be implemented here
        pass

    def upload_image(self):
        """Handle image upload"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]

        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=file_types
        )

        if file_path:
            try:
                # Load and display the image
                self.image_path = file_path
                self.original_image = cv2.imread(file_path)
                self.current_image = self.original_image.copy()

                # Display the image
                self.display_image(self.current_image)

                # Enable detect button
                self.detect_btn.config(state='normal')

                # Update results
                self.results_label.config(
                    text=f"Image loaded: {os.path.basename(file_path)}")

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to load image: {str(e)}")

    def display_image(self, cv_image):
        """Display OpenCV image in tkinter"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Resize image to fit in the display area
        height, width = rgb_image.shape[:2]
        max_width, max_height = 600, 400

        if width > max_width or height > max_height:
            ratio = min(max_width/width, max_height/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))

        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)

        # Update the label
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference

    def detect_buses(self):
        """Detect bus in the uploaded image using trained YOLO model"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return

        if self.model is None:
            messagebox.showerror(
                "Error", "YOLO model not loaded. Please check the model file.")
            return

        try:
            # Update results to show processing
            self.results_label.config(
                text="Processing image... Please wait", fg='blue')
            self.root.update()

            # Run inference on the image
            results = self.model(self.current_image)

            # Draw bounding boxes on the image
            result_image = self.current_image.copy()
            bus_count = 0

            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        # Get class name
                        if class_id < len(self.classes):
                            class_name = self.classes[class_id]
                        else:
                            class_name = f"Class_{class_id}"

                        # Check if confidence is above threshold
                        if confidence > 0.5:  # Confidence threshold
                            bus_count += 1

                            # Draw bounding box
                            cv2.rectangle(result_image, (x1, y1),
                                          (x2, y2), (0, 255, 0), 2)

                            # Add label with class name and confidence
                            label = f"{class_name}: {confidence:.2f}"
                            label_size = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                            # Draw label background
                            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                                          (x1 + label_size[0], y1), (0, 255, 0), -1)

                            # Draw label text
                            cv2.putText(result_image, label, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Display the result
            self.display_image(result_image)

            # Update results
            if bus_count > 0:
                detection_text = f"Detection complete! Found {bus_count} object(s) in the image"
                if len(self.classes) == 1:
                    detection_text = f"Detection complete! Found {bus_count} bus(es) in the image"
                self.results_label.config(text=detection_text, fg='green')
                # Crop detected buses and process them
                self.main_processes(results)
            else:
                self.results_label.config(
                    text="Detection complete! No objects found above confidence threshold",
                    fg='orange'
                )

        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.results_label.config(text="Detection failed", fg='red')

    def main_processes(self, results):
        """Crop detected objects and pass them to processing function"""
        if self.original_image is None:
            return

        cropped_images = []
        detection_info = []

        try:
            # Process each detection result
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        # Only process detections above confidence threshold
                        if confidence > 0.5:
                            # Get class name
                            if class_id < len(self.classes):
                                class_name = self.classes[class_id]
                            else:
                                class_name = f"Class_{class_id}"

                            # Crop the detected object from original image
                            cropped_img = self.crop_detection(x1, y1, x2, y2)

                            if cropped_img is not None:

                                # Append info
                                # Remember to modify to two classes after change of model
                                cropped_images.append(cropped_img)
                                detection_info.append({
                                    'class_name': class_name,
                                    'confidence': confidence,
                                    'bbox': (x1, y1, x2, y2),
                                    'crop_id': len(cropped_images) - 1
                                })

            print(detection_info)
            # print("\n")
            best_cropped_images = self.handle_best_detections(
                cropped_images, detection_info)
            ocr_results = self.ocr_cropped_images(best_cropped_images)
            print(ocr_results)
            if ocr_results is not None:
                if len(ocr_results) == 2:
                    sim_search = self.similarity_search(ocr_results)
                    self.speak(sim_search)
                else:
                    # only bus number
                    self.speak(ocr_results)

        except Exception as e:
            print(f"Error in main_processes: {e}")

    def handle_best_detections(self, cropped_images, detection_info):

        best_cropped_images = []

        def get_best_entry(entries, class_name):
            filtered = [
                info for info in entries if info['class_name'].lower() == class_name]
            if not filtered:
                print(f"No '{class_name}' detections above threshold.")
                return None
            # print("Hello 1")
            return max(filtered, key=lambda info: info['confidence'])

        # print("Hello 2")

        # Get best detections
        best_destination = get_best_entry(detection_info, "destination")
        # print("Hello 3")

        best_bus_number = get_best_entry(detection_info, "bus_number")

        # print("Best destination:\n", best_destination)
        # print(best_destination.keys())
        # print(best_destination['crop_id'])
        output_dir = "tk_cropped_outputs"

        if best_destination is None:
            pass
        else:
            best_destination_image = cropped_images[best_destination['crop_id']]
            destination_path = os.path.join(
                output_dir, f"best_cropped_{best_destination['class_name']}.jpg")
            cv2.imwrite(destination_path, best_destination_image)
            best_cropped_images.append(best_destination_image)

        if best_bus_number is None:
            pass
        else:
            best_bus_number_image = cropped_images[best_bus_number['crop_id']]
            bus_number_path = os.path.join(
                output_dir, f"best_cropped_{best_bus_number['class_name']}.jpg")
            cv2.imwrite(bus_number_path, best_bus_number_image)
            best_cropped_images.append(best_bus_number_image)

        # print(best_cropped_images)
        return best_cropped_images

    def ocr_cropped_images(self, best_cropped_images: list):
        # if there is only bus number or destination, do not go to next step of similarity search
        ocr_results = []
        print(len(best_cropped_images))
        print(best_cropped_images is None)

        if len(best_cropped_images) == 0:
            TTS().fail()
        else:
            for idx, img in enumerate(best_cropped_images):
                try:
                    # Convert numpy array to PIL Image for TrOCR
                    if isinstance(img, np.ndarray):
                        # Convert numpy array to PIL Image
                        if img.dtype != np.uint8:
                            # If image is normalized
                            img = (img * 255).astype(np.uint8)

                        # Handle different array shapes
                        if len(img.shape) == 3:
                            pil_image = Image.fromarray(img)
                        else:
                            pil_image = Image.fromarray(img).convert('RGB')
                    else:
                        pil_image = img  # Assume it's already a PIL Image

                    # Save temporarily or pass directly to TrOCR
                    # Option 1: Save temp image and use file path
                    temp_path = f"temp_img_{idx}.png"
                    pil_image.save(temp_path)

                    # Use TrOCR inference
                    extracted_text = self.trocr.inference(temp_path)

                    # Clean up temp file
                    import os
                    os.remove(temp_path)

                    print("EACH OCR RESULT:", extracted_text)
                    ocr_results.append(extracted_text)

                except Exception as e:
                    print(f"Error processing image {idx}: {e}")

            if len(ocr_results) == 1 and not (ocr_results[0].isnumeric()):
                TTS().fail()
                return
            if '' in ocr_results:
                TTS().fail()
                return
            return ocr_results

    def similarity_search(self, ocr_results: list):
        print("OCR RESULTS: ", ocr_results)
        if len(ocr_results) == 2:
            destination = ocr_results[0]
            bus_number = ocr_results[1]
            result_list = self.ss.extract_wiki_origin_and_destination(
                bus_number)
            return self.ss.match_ocr_text(destination, result_list)

    def speak(self, results):
        print("SPEAK RESULTS: ", results)
        if len(results) == 1:
            TTS().speak_bus_number(results)
        else:
            bus_number, destination = results
            print(destination.lower())
            if destination.lower() == "route not found":
                TTS().fail_to_read_number()
            else:
                TTS().speak(bus_number, destination)

    def crop_detection(self, x1, y1, x2, y2):
        """Crop a single detection from the original image"""
        try:
            # Ensure coordinates are within image bounds
            height, width = self.original_image.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Ensure valid bounding box
            if x2 <= x1 or y2 <= y1:
                return None

            # Add some padding around the detection (optional)
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            # Crop the image
            cropped_img = self.original_image[y1:y2, x1:x2].copy()
            return cropped_img

        except Exception as e:
            print(f"Error cropping detection: {e}")
            return None

    def clear_image(self):
        """Clear the current image and reset the interface"""
        if self.current_page != "image":
            return

        self.current_image = None
        self.original_image = None
        self.image_path = None

        # Reset image display
        self.image_label.config(
            image="",
            text="No image uploaded\nClick 'Upload Image' to get started"
        )
        self.image_label.image = None

        # Disable detect button
        self.detect_btn.config(state='disabled')

        # Reset results
        self.results_label.config(
            text="Detection results will appear here", fg='#666666')


def main():
    root = tk.Tk()
    app = BusDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

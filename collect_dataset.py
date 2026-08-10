#!/usr/bin/env python3
"""
Drone Gesture Dataset Collector
===============================
Author: Senior Python Developer & AI Engineer
Description: A high-performance Python script to collect hand gesture images 
             for YOLOv8 training. Implements asynchronous background saving
             to prevent UI stuttering, and renders a premium semi-transparent
             HUD overlay directly on the frame with visual gesture guides and icons.
Technology Stack: Python 3.11+, OpenCV (cv2)
"""

import os
import sys
import time
import queue
import threading
from typing import Dict, Tuple, Optional
import cv2

# Try to import Pillow for premium HUD rendering with unicode emojis & Vietnamese text
try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Configuration Constants
DATASET_DIR = "dataset"
CLASSES: Dict[int, str] = {
    1: "Forward",
    2: "Backward",
    3: "Right",
    4: "Left",
    5: "Takeoff",
    6: "Land",
    7: "Up",
    8: "Down",
    9: "Hover"
}
AUTO_CAPTURE_INTERVAL = 0.5  # Time interval for auto-capture in seconds

# Gesture Guideline Information
GESTURE_DETAILS: Dict[str, Dict[str, str]] = {
    "Forward": {
        "emoji": "☝️",
        "name": "1 ngón",
        "desc": "Bay tiến",
        "ascii_icon": "1 Ngon",
        "ascii_desc": "Bay tien"
    },
    "Backward": {
        "emoji": "✌️",
        "name": "2 ngón",
        "desc": "Bay lùi",
        "ascii_icon": "2 Ngon",
        "ascii_desc": "Bay lui"
    },
    "Right": {
        "emoji": "🤟",
        "name": "3 ngón",
        "desc": "Bay sang phải",
        "ascii_icon": "3 Ngon",
        "ascii_desc": "Bay sang phai"
    },
    "Left": {
        "emoji": "🖐",
        "name": "4 ngón",
        "desc": "Bay sang trái",
        "ascii_icon": "4 Ngon",
        "ascii_desc": "Bay sang trai"
    },
    "Takeoff": {
        "emoji": "🖐️",
        "name": "Bàn tay mở 5 ngón",
        "desc": "Cất cánh (Takeoff)",
        "ascii_icon": "5 Ngon Mo",
        "ascii_desc": "Cat canh"
    },
    "Land": {
        "emoji": "✊",
        "name": "Nắm tay",
        "desc": "Hạ cánh (Land)",
        "ascii_icon": "Nam tay",
        "ascii_desc": "Ha canh"
    },
    "Up": {
        "emoji": "👍",
        "name": "Ngón cái hướng lên",
        "desc": "Tăng độ cao",
        "ascii_icon": "Ngon cai len",
        "ascii_desc": "Tang do cao"
    },
    "Down": {
        "emoji": "👎",
        "name": "Ngón cái hướng xuống",
        "desc": "Giảm độ cao",
        "ascii_icon": "Ngon cai xuong",
        "ascii_desc": "Giam do cao"
    },
    "Hover": {
        "emoji": "🤙",
        "name": "Ký hiệu Shaka",
        "desc": "Dừng khẩn cấp / Hover",
        "ascii_icon": "Shaka (🤙)",
        "ascii_desc": "Dung khan cap / Hover"
    }
}


class ImageSaver(threading.Thread):
    """
    A consumer thread that processes image-saving tasks from a queue.
    This prevents the main camera-capture loop from lagging due to I/O operations.
    """
    def __init__(self):
        super().__init__()
        self.queue: queue.Queue = queue.Queue()
        self.running: bool = True
        self.daemon: bool = True

    def run(self) -> None:
        while self.running or not self.queue.empty():
            try:
                # Use a short timeout so we periodically check if self.running is False
                task = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame, filepath = task
            try:
                # Save frame at maximum quality (JPEG quality = 100)
                success = cv2.imwrite(
                    filepath, 
                    frame, 
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                )
                if success:
                    print(f"[SUCCESS] Saved: {filepath}")
                else:
                    print(f"[ERROR] Failed to write file: {filepath}", file=sys.stderr)
            except Exception as e:
                print(f"[EXCEPTION] Failed to save {filepath}: {e}", file=sys.stderr)
            finally:
                self.queue.task_done()

    def save_image(self, frame: cv2.Mat, filepath: str) -> None:
        """Enqueue an image saving task. We copy the frame to avoid mutation issues."""
        self.queue.put((frame.copy(), filepath))

    def stop(self) -> None:
        """Signifies the thread to stop processing and exit."""
        self.running = False


def setup_directories(base_dir: str, classes: Dict[int, str]) -> None:
    """Create the base dataset folder and class-specific subdirectories if they do not exist."""
    try:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            print(f"[INFO] Created base dataset folder: '{base_dir}'")
        
        for class_name in classes.values():
            class_path = os.path.join(base_dir, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path, exist_ok=True)
                print(f"[INFO] Created class folder: '{class_path}'")
    except Exception as e:
        print(f"[CRITICAL] Error creating directories: {e}", file=sys.stderr)
        sys.exit(1)


def get_next_image_index(class_dir: str) -> int:
    """
    Scans the class folder to find the next incrementing index.
    Looks for files named 'XXXXXX.jpg' where XXXXXX is a 6-digit integer.
    """
    if not os.path.exists(class_dir):
        return 1
    
    indices = []
    for filename in os.listdir(class_dir):
        if filename.lower().endswith(".jpg"):
            name_part = filename[:-4]
            if name_part.isdigit() and len(name_part) == 6:
                indices.append(int(name_part))
                
    return max(indices) + 1 if indices else 1


def get_saved_images_count(class_dir: str) -> int:
    """Returns the total number of JPEG images inside the class directory."""
    if not os.path.exists(class_dir):
        return 0
    return sum(1 for f in os.listdir(class_dir) if f.lower().endswith(".jpg"))


def reset_class_directory(class_dir: str) -> None:
    """Deletes all .jpg images inside the given directory to reset the counter."""
    if not os.path.exists(class_dir):
        return
    
    print(f"[INFO] Resetting directory: '{class_dir}'...")
    deleted_count = 0
    for filename in os.listdir(class_dir):
        if filename.lower().endswith(".jpg"):
            filepath = os.path.join(class_dir, filename)
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"[ERROR] Could not delete {filepath}: {e}", file=sys.stderr)
                
    print(f"[SUCCESS] Deleted {deleted_count} image(s) from '{class_dir}'")


def draw_hud(
    frame: cv2.Mat,
    current_class_name: str,
    images_saved: int,
    auto_capture_on: bool,
    countdown: float,
    fps: float,
    camera_id: int
) -> cv2.Mat:
    """Renders a premium, semi-transparent HUD overlay on the frame with visual guides."""
    h, w, _ = frame.shape
    
    # Define HUD Panel geometry (slightly wider for dual languages)
    panel_w = 280
    overlay = frame.copy()
    
    # Draw dark overlay panel for the side panel
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (25, 25, 25), -1)
    
    # Visual border line for the panel
    cv2.line(overlay, (panel_w, 0), (panel_w, h), (70, 70, 70), 1)
    
    # Blending the transparent side panel with the webcam stream (75% panel, 25% original stream)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    
    if HAS_PIL:
        # PIL text rendering allows for beautiful Unicode characters (Vietnamese & Emojis)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Load Windows fonts or fallback to default
        try:
            win_font_dir = "C:\\Windows\\Fonts"
            font_title = ImageFont.truetype(os.path.join(win_font_dir, "arialbd.ttf"), 13)
            font_text = ImageFont.truetype(os.path.join(win_font_dir, "arial.ttf"), 12)
            font_desc = ImageFont.truetype(os.path.join(win_font_dir, "arialbd.ttf"), 13)
            font_val = ImageFont.truetype(os.path.join(win_font_dir, "arialbd.ttf"), 17)
            font_key = ImageFont.truetype(os.path.join(win_font_dir, "arial.ttf"), 11)
            # Try to load emoji font for proper color emojis, otherwise fallback to Arial
            font_emoji = ImageFont.truetype(os.path.join(win_font_dir, "seguiemj.ttf"), 15)
        except Exception:
            font_title = font_text = font_desc = font_val = font_key = font_emoji = ImageFont.load_default()
            
        color_title = (255, 215, 0)       # Gold
        color_text = (240, 240, 240)       # Near White
        color_val = (127, 255, 0)         # Spring Green
        color_alert_on = (255, 50, 50)     # Vibrant Red
        color_alert_off = (140, 140, 140)  # Neutral Gray
        color_key = (135, 206, 235)       # Sky Blue
        
        y_offset = 20
        # 1. Header Title
        draw.text((15, y_offset), "DRONE GESTURE COLLECTOR", font=font_title, fill=color_title)
        draw.line([(15, y_offset + 22), (panel_w - 15, y_offset + 22)], fill=(90, 90, 90), width=1)
        
        # 2. Current Class Info
        y_offset += 35
        draw.text((15, y_offset), "Lớp hiện tại (Class):", font=font_text, fill=color_text)
        draw.text((15, y_offset + 16), current_class_name, font=font_val, fill=color_val)
        
        # 3. Gesture Guide Info
        y_offset += 45
        draw.text((15, y_offset), "Cử chỉ yêu cầu (Gesture):", font=font_text, fill=color_text)
        
        gesture_info = GESTURE_DETAILS.get(current_class_name, {})
        emoji_char = gesture_info.get("emoji", "")
        gesture_name = gesture_info.get("name", "")
        gesture_desc = gesture_info.get("desc", "")
        
        # Draw emoji using the emoji font and text using normal font
        draw.text((15, y_offset + 18), f"{emoji_char} {gesture_name}", font=font_desc, fill=color_title)
        draw.text((15, y_offset + 36), f"→ {gesture_desc}", font=font_text, fill=color_val)
        
        # 4. Images Saved Info
        y_offset += 65
        draw.text((15, y_offset), "Số ảnh đã lưu (Saved):", font=font_text, fill=color_text)
        draw.text((15, y_offset + 16), str(images_saved), font=font_val, fill=color_val)
        
        # 5. Auto Capture Info
        y_offset += 45
        draw.text((15, y_offset), "Chế độ Auto Capture:", font=font_text, fill=color_text)
        status_str = "ON" if auto_capture_on else "OFF"
        status_color = color_alert_on if auto_capture_on else color_alert_off
        draw.text((15, y_offset + 16), status_str, font=font_val, fill=status_color)
        
        if auto_capture_on:
            blink = int(time.time() * 2) % 2
            if blink == 0:
                draw.ellipse([panel_w - 35, y_offset + 20, panel_w - 23, y_offset + 32], fill=(255, 0, 0))
            draw.text((15, y_offset + 38), f"Chụp tiếp sau: {countdown:.1f}s", font=font_text, fill=color_title)
            y_offset += 18
            
        # 6. Keyboard Shortcut Guide
        y_offset += 45
        draw.text((15, y_offset), "PHÍM TẮT ĐIỀU KHIỂN", font=font_title, fill=color_title)
        draw.line([(15, y_offset + 20), (panel_w - 15, y_offset + 20)], fill=(90, 90, 90), width=1)
        
        shortcuts = [
            ("1-9", "Chọn Lớp cử chỉ"),
            ("S", "Chụp & Lưu 1 ảnh"),
            ("A", "Bật/Tắt Tự động"),
            ("R", "Xóa ảnh lớp này"),
            ("C", "Chuyển Camera"),
            ("Q/ESC", "Thoát chương trình")
        ]
        
        y_offset += 24
        for key, desc in shortcuts:
            draw.text((15, y_offset), f"{key}:", font=font_key, fill=color_key)
            draw.text((85, y_offset), desc, font=font_key, fill=color_text)
            y_offset += 18
            
        # Convert PIL Image back to OpenCV (BGR)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    else:
        # Fallback to pure OpenCV text rendering if Pillow is not available
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        color_title = (0, 215, 255)       # Gold / Warm Cyan
        color_text = (240, 240, 240)       # Near White
        color_val = (0, 255, 127)         # Spring Green
        color_alert_on = (50, 50, 255)     # Vibrant Red
        color_alert_off = (140, 140, 140)  # Neutral Gray
        color_key = (235, 206, 135)       # Sky Blue
        
        y_offset = 30
        cv2.putText(frame, "DRONE GESTURE COLLECTOR", (15, y_offset), font, 0.52, color_title, 2, cv2.LINE_AA)
        cv2.line(frame, (15, y_offset + 8), (panel_w - 15, y_offset + 8), (90, 90, 90), 1)
        
        # Current Class Info
        y_offset += 45
        cv2.putText(frame, "Current Class:", (15, y_offset), font, font_scale, color_text, thickness, cv2.LINE_AA)
        cv2.putText(frame, current_class_name, (15, y_offset + 22), font, 0.65, color_val, 2, cv2.LINE_AA)
        
        # Gesture Info
        y_offset += 55
        cv2.putText(frame, "Gesture Guide:", (15, y_offset), font, font_scale, color_text, thickness, cv2.LINE_AA)
        gesture_info = GESTURE_DETAILS.get(current_class_name, {})
        ascii_icon = gesture_info.get("ascii_icon", "")
        ascii_desc = gesture_info.get("ascii_desc", "")
        cv2.putText(frame, f"{ascii_icon}", (15, y_offset + 18), font, 0.5, color_title, 1, cv2.LINE_AA)
        cv2.putText(frame, f"-> {ascii_desc}", (15, y_offset + 32), font, 0.45, color_val, 1, cv2.LINE_AA)
        
        # Images Saved Info
        y_offset += 55
        cv2.putText(frame, "Images Saved:", (15, y_offset), font, font_scale, color_text, thickness, cv2.LINE_AA)
        cv2.putText(frame, str(images_saved), (15, y_offset + 22), font, 0.65, color_val, 2, cv2.LINE_AA)
        
        # Auto Capture Info
        y_offset += 55
        cv2.putText(frame, "Auto Capture:", (15, y_offset), font, font_scale, color_text, thickness, cv2.LINE_AA)
        status_str = "ON" if auto_capture_on else "OFF"
        status_color = color_alert_on if auto_capture_on else color_alert_off
        cv2.putText(frame, status_str, (15, y_offset + 22), font, 0.65, status_color, 2, cv2.LINE_AA)
        
        if auto_capture_on:
            blink = int(time.time() * 2) % 2
            if blink == 0:
                cv2.circle(frame, (panel_w - 35, y_offset + 16), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"Next save in: {countdown:.1f}s", (15, y_offset + 40), font, 0.45, color_title, thickness, cv2.LINE_AA)
            y_offset += 20
            
        # Shortcuts Guide
        y_offset += 55
        cv2.putText(frame, "KEYBOARD SHORTCUTS", (15, y_offset), font, 0.48, color_title, 2, cv2.LINE_AA)
        cv2.line(frame, (15, y_offset + 6), (panel_w - 15, y_offset + 6), (90, 90, 90), 1)
        
        shortcuts = [
            ("1-9", "Select Class"),
            ("S", "Save Image"),
            ("A", "Toggle Auto"),
            ("R", "Reset Counter"),
            ("C", "Switch Camera"),
            ("Q/ESC", "Quit Program")
        ]
        
        y_offset += 22
        for key, desc in shortcuts:
            cv2.putText(frame, f"{key}:", (15, y_offset), font, 0.42, color_key, 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (85, y_offset), font, 0.42, color_text, 1, cv2.LINE_AA)
            y_offset += 18
            
    # Draw Upper Right Overlay Info (FPS & Cam ID)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps_info = f"FPS: {fps:.1f} | Cam: {camera_id}"
    text_size = cv2.getTextSize(fps_info, font, 0.45, 1)[0]
    box_padding = 6
    
    cv2.rectangle(
        frame,
        (w - text_size[0] - box_padding * 2 - 10, 10),
        (w - 10, 10 + text_size[1] + box_padding * 2),
        (25, 25, 25),
        -1
    )
    cv2.rectangle(
        frame,
        (w - text_size[0] - box_padding * 2 - 10, 10),
        (w - 10, 10 + text_size[1] + box_padding * 2),
        (80, 80, 80),
        1
    )
    cv2.putText(
        frame, 
        fps_info, 
        (w - text_size[0] - box_padding - 10, 10 + text_size[1] + box_padding), 
        font, 
        0.45, 
        (0, 255, 127), 
        1, 
        cv2.LINE_AA
    )
    return frame


def initialize_camera(camera_idx: int) -> Optional[cv2.VideoCapture]:
    """Attempts to initialize a camera capture stream using DSHOW backend (and falls back to standard)."""
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_idx)
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap
    return None


def main() -> None:
    print("==================================================")
    print("      DRONE GESTURE DATASET COLLECTOR CAPTURE     ")
    print("==================================================")
    
    # Step 1: Directory Setup
    setup_directories(DATASET_DIR, CLASSES)
    
    # State tracking variables
    current_class_id = 1
    auto_capture_on = False
    camera_idx = 0
    
    # Cache to track filenames & count without repeated disk hits in main loop
    saved_counts = {}
    next_write_indices = {}
    for cid, cname in CLASSES.items():
        class_dir = os.path.join(DATASET_DIR, cname)
        saved_counts[cid] = get_saved_images_count(class_dir)
        next_write_indices[cid] = get_next_image_index(class_dir)
        
    # Step 2: Initialize Background Image Saver Thread
    saver_thread = ImageSaver()
    saver_thread.start()
    
    # Step 3: Initialize Video Stream
    cap = initialize_camera(camera_idx)
    if cap is None:
        print(f"[CRITICAL] Could not open camera at index {camera_idx}.", file=sys.stderr)
        saver_thread.stop()
        saver_thread.join()
        sys.exit(1)
        
    print(f"[INFO] Camera {camera_idx} successfully initialized.")
    print("[INFO] Press 'Q' or 'ESC' on the OpenCV Window to Quit.")
    
    # Performance & Timing variables
    fps = 0.0
    alpha_fps = 0.9  # Exponential moving average factor for FPS
    last_frame_time = time.time()
    
    last_auto_capture_time = time.time()
    
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame capture returned empty. Camera may be disconnected.")
                time.sleep(0.03)
                continue
            
            # FPS Calculation (Smoothed out using exponential decay)
            curr_time = time.time()
            dt = curr_time - last_frame_time
            last_frame_time = curr_time
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = (alpha_fps * fps) + ((1.0 - alpha_fps) * instant_fps)
                
            # Class path definition
            current_class_name = CLASSES[current_class_id]
            class_dir = os.path.join(DATASET_DIR, current_class_name)
            
            # Auto Capture logic
            countdown = AUTO_CAPTURE_INTERVAL
            if auto_capture_on:
                elapsed = time.time() - last_auto_capture_time
                countdown = max(0.0, AUTO_CAPTURE_INTERVAL - elapsed)
                
                if elapsed >= AUTO_CAPTURE_INTERVAL:
                    # Capture frame
                    filename = f"{next_write_indices[current_class_id]:06d}.jpg"
                    filepath = os.path.join(class_dir, filename)
                    
                    # Queue saving task
                    saver_thread.save_image(frame, filepath)
                    
                    # Update local state tracking
                    next_write_indices[current_class_id] += 1
                    saved_counts[current_class_id] += 1
                    
                    # Reset timer
                    last_auto_capture_time = time.time()
                    countdown = AUTO_CAPTURE_INTERVAL
            
            # Draw overlay HUD
            display_frame = draw_hud(
                frame.copy(), 
                current_class_name, 
                saved_counts[current_class_id], 
                auto_capture_on, 
                countdown, 
                fps, 
                camera_idx
            )
            
            # Show OpenCV preview window
            cv2.imshow("Drone Gesture Collector", display_frame)
            
            # Capture User Input
            key = cv2.waitKey(1) & 0xFF
            
            # Exit controls (ESC or Q)
            if key == 27 or key == ord('q') or key == ord('Q'):
                print("[INFO] Exiting program...")
                break
                
            # Class switching (1 to 9)
            elif ord('1') <= key <= ord('9'):
                new_class_id = int(chr(key))
                if new_class_id in CLASSES:
                    current_class_id = new_class_id
                    if auto_capture_on:
                        last_auto_capture_time = time.time()
                    print(f"[INFO] Switched to class {current_class_id}: '{CLASSES[current_class_id]}'")
                    
            # Manual Save (S)
            elif key == ord('s') or key == ord('S'):
                filename = f"{next_write_indices[current_class_id]:06d}.jpg"
                filepath = os.path.join(class_dir, filename)
                
                # Queue manual save
                saver_thread.save_image(frame, filepath)
                
                # Update counters
                next_write_indices[current_class_id] += 1
                saved_counts[current_class_id] += 1
                
            # Toggle Auto Capture (A)
            elif key == ord('a') or key == ord('A'):
                auto_capture_on = not auto_capture_on
                if auto_capture_on:
                    last_auto_capture_time = time.time()
                print(f"[INFO] Auto Capture toggled to: {'ON' if auto_capture_on else 'OFF'}")
                
            # Reset Class Counter (R)
            elif key == ord('r') or key == ord('R'):
                reset_class_directory(class_dir)
                saved_counts[current_class_id] = 0
                next_write_indices[current_class_id] = 1
                if auto_capture_on:
                    last_auto_capture_time = time.time()
                    
            # Switch Camera Device (C)
            elif key == ord('c') or key == ord('C'):
                auto_capture_on = False
                next_idx = camera_idx + 1
                print(f"[INFO] Attempting to open camera index {next_idx}...")
                
                new_cap = initialize_camera(next_idx)
                if new_cap is not None:
                    cap.release()
                    cap = new_cap
                    camera_idx = next_idx
                    print(f"[SUCCESS] Switched to camera index {camera_idx}")
                else:
                    print(f"[WARNING] Camera {next_idx} failed. Testing index 0...")
                    new_cap = initialize_camera(0)
                    if new_cap is not None:
                        cap.release()
                        cap = new_cap
                        camera_idx = 0
                        print(f"[SUCCESS] Switched back to camera index 0")
                    else:
                        print("[ERROR] Failed to switch camera. Keeping current stream.")
                        
    except KeyboardInterrupt:
        print("[INFO] Interrupted by keyboard (Ctrl+C)")
    finally:
        # Cleanup Resources
        print("[INFO] Releasing camera...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        
        print("[INFO] Waiting for background saver to finish writing pending frames...")
        saver_thread.stop()
        saver_thread.join()
        print("[INFO] Setup complete. Dataset saved successfully!")


if __name__ == "__main__":
    main()

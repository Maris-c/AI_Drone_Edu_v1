import os
import sys
import cv2
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dist" / "dataset"
OUTPUT_DIR = BASE_DIR / "dist" / "test_results"
MODEL_PATH_ROOT = BASE_DIR / "best.pt"
MODEL_PATH_RUNS = BASE_DIR / "train_yolo" / "runs" / "detect" / "train-2" / "weights" / "best.pt"

# Class name mapping between folder names and model labels
FOLDER_TO_CLASS = {
    "Land": "0-Land",
    "Forward": "1-Forward",
    "Backward": "2-Backward",
    "Right": "3-Right",
    "Left": "4-Left",
    "Takeoff": "5-Takeoff",
    "Down": "Down",
    "Hover": "Hover",
    "Up": "Up"
}

def print_header(title):
    print("=" * 65)
    print(f" {title} ".center(65, "="))
    print("=" * 65)

def check_dataset_integrity():
    print_header("1. KIEM TRA DU LIEU ANH (DATASET INTEGRITY)")
    
    if not DATASET_DIR.exists():
        print(f"[ERR] Khong tim thay thu muc dataset tai: {DATASET_DIR}")
        print("Vui long chay file collect_dataset.exe truoc de thu thap anh.")
        return False

    classes = list(FOLDER_TO_CLASS.keys())
    all_ok = True
    
    print(f"{'Thu muc cu chi':<18} | {'So luong anh':<12} | {'Do phan giai':<15} | {'Trang thai':<10}")
    print("-" * 68)
    
    for cls in classes:
        cls_dir = DATASET_DIR / cls
        if not cls_dir.exists():
            print(f"{cls:<18} | {'0 (Khong thay)':<12} | {'N/A':<15} | [ERR] THIEU THU MUC")
            all_ok = False
            continue
            
        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        img_count = len(images)
        
        if img_count == 0:
            print(f"{cls:<18} | {img_count:<12} | {'N/A':<15} | [ERR] KHONG CO ANH")
            all_ok = False
            continue
            
        # Check integrity of the first image as a sample
        sample_img_path = images[0]
        try:
            img = cv2.imread(str(sample_img_path))
            if img is not None:
                h, w, c = img.shape
                res_str = f"{w}x{h}"
                status = "[OK]"
            else:
                res_str = "N/A"
                status = "[ERR] HONG FILE"
                all_ok = False
        except Exception as e:
            res_str = "N/A"
            status = f"[ERR] LOI ({str(e)})"
            all_ok = False
            
        # Warn if too few images (less than 10 is too few according to guidelines)
        if img_count < 10:
            status += " ([WARN] It anh)"
            
        print(f"{cls:<18} | {img_count:<12} | {res_str:<15} | {status}")
        
    print("-" * 68)
    if all_ok:
        print("[SUCCESS] Toan bo thu muc anh hop le va san sang!")
    else:
        print("[WARN] Co loi hoac thieu thu muc anh. Hay kiem tra lai cac muc [ERR].")
    return all_ok

def run_model_inference():
    print_header("2. KIEM TRA MO HINH AI (MODEL INFERENCE TEST)")
    
    # Check model weights
    model_path = None
    if MODEL_PATH_ROOT.exists():
        model_path = MODEL_PATH_ROOT
    elif MODEL_PATH_RUNS.exists():
        model_path = MODEL_PATH_RUNS
    
    if not model_path:
        print("[ERR] Khong tim thay file trong so mo hinh (best.pt).")
        print(f"Da tim kiem tai:\n- {MODEL_PATH_ROOT}\n- {MODEL_PATH_RUNS}")
        print("Vui long huan luyen mo hinh truoc khi chay kiem tra.")
        return
        
    print(f"[INFO] Dang tai mo hinh tu: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERR] Thu vien 'ultralytics' chua duoc cai dat. Vui long chay lenh: pip install ultralytics")
        return

    try:
        model = YOLO(str(model_path))
        print("[OK] Tai mo hinh thanh cong!")
    except Exception as e:
        print(f"[ERR] Loi khi tai mo hinh: {e}")
        return

    # Create output directory for test predictions
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[PATH] Ket qua du doan se duoc luu tai: {OUTPUT_DIR}\n")

    total_images = 0
    correct_detections = 0
    total_detected = 0
    
    print(f"{'Thu muc cu chi':<18} | {'Tong anh':<10} | {'Da phat hien':<12} | {'Dung cu chi':<12} | {'Do chinh xac':<12}")
    print("-" * 72)
    
    for folder_name, model_class_name in FOLDER_TO_CLASS.items():
        cls_dir = DATASET_DIR / folder_name
        if not cls_dir.exists():
            continue
            
        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not images:
            continue
            
        folder_total = 0
        folder_detected = 0
        folder_correct = 0
        
        # Create folder for saving annotated images for this class
        cls_output_dir = OUTPUT_DIR / folder_name
        cls_output_dir.mkdir(exist_ok=True)
        
        for img_path in images:
            folder_total += 1
            total_images += 1
            
            # Predict
            results = model(str(img_path), verbose=False)
            
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                # Check if anything was detected in the image
                if len(boxes) > 0:
                    folder_detected += 1
                    total_detected += 1
                    
                    # Check if the highest confidence detection matches the expected label
                    best_box_idx = boxes.conf.argmax()
                    pred_class_idx = int(boxes.cls[best_box_idx])
                    pred_class_name = model.names[pred_class_idx]
                    
                    # Save annotated image
                    annotated_frame = result.plot()
                    cv2.imwrite(str(cls_output_dir / img_path.name), annotated_frame)
                    
                    # Verify if prediction matches folder class
                    if pred_class_name == model_class_name:
                        folder_correct += 1
                        correct_detections += 1
                        
        acc = (folder_correct / folder_total) * 100 if folder_total > 0 else 0
        print(f"{folder_name:<18} | {folder_total:<10} | {folder_detected:<12} | {folder_correct:<12} | {acc:.1f}%")
        
    print("-" * 72)
    total_acc = (correct_detections / total_images) * 100 if total_images > 0 else 0
    print(f"{'TONG CONG':<18} | {total_images:<10} | {total_detected:<12} | {correct_detections:<12} | {total_acc:.1f}%")
    print(f"\n[NOTE] Ban co the mo thu muc '{OUTPUT_DIR}' de xem cac anh da duoc ve khung nhan dien (bounding box).")

if __name__ == "__main__":
    dataset_ok = check_dataset_integrity()
    print()
    if dataset_ok:
        run_model_inference()

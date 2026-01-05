import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import cv2
import numpy as np
from model.flol import create_model
from options.options import parse
from ultralytics import YOLO  

def pad_tensor(tensor, multiple=8):
    '''Thêm viền (pad) để kích thước ảnh chia hết cho 8'''
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang chạy trên thiết bị: {device} ---")

    # --- KHỞI TẠO MODEL 1: FLOL ---
    print("⏳ Đang tải mô hình FLOL...")
    flol_model = create_model()
    weights_path = opt['settings']['weight']
    checkpoint = torch.load(weights_path, map_location=device)
    flol_model.load_state_dict(checkpoint['params'])
    flol_model.to(device)
    flol_model.eval() 
    print("Tải FLOL thành công!")

    # --- KHỞI TẠO MODEL 2: YOLO  ---
    print("⏳ Đang tải mô hình YOLO...")
    try:
        yolo_model = YOLO("yolo11n.pt") 
        print("Tải YOLO thành công!")
    except Exception as e:
        print(f"Lỗi không tìm thấy file best.pt: {e}")
        return

    # Webcam 
    cap = cv2.VideoCapture(1) 
    if not cap.isOpened():
        print("Không thể mở Webcam!")
        return

    to_tensor = transforms.ToTensor()
    print("Đang chạy Real-time. Nhấn phím 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Resize nhẹ nếu thấy lag 
        # frame = cv2.resize(frame, (640, 480))
        
        # --- BƯỚC 1: TIỀN XỬ LÝ & FLOL ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = to_tensor(img_rgb).unsqueeze(0).to(device)
        
        # Padding & Inference
        _, _, H, W = img_tensor.shape
        img_padded = pad_tensor(img_tensor)

        with torch.no_grad():
            output = flol_model(img_padded)

        # Hậu xử lý FLOL -> Ra ảnh sáng
        output = torch.clamp(output, 0., 1.)
        output = output[:, :, :H, :W] # Cắt bỏ padding
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Chuyển về ảnh chuẩn OpenCV (BGR)
        output_bgr = (output_np * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR)

        # --- BƯỚC 2: CHẠY YOLO TRÊN ẢNH ĐÃ LÀM SÁNG ---
        # conf=0.5 :chắc chắn trên 50% mới vẽ khung
        results = yolo_model(output_bgr, verbose=False, conf=0.5) 
        
        # Lấy ảnh đã được YOLO vẽ khung sẵn
        yolo_frame = results[0].plot()

        # --- HIỂN THỊ ---
        if frame.shape == yolo_frame.shape:
             combined = np.hstack((frame, yolo_frame))
        else:
             # Phòng hờ trường hợp kích thước lệch nhau thì chỉ hiện kết quả
             combined = yolo_frame

        cv2.putText(combined, "Original Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "FLOL Detect", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Project Demo: Low Light Enhancement + Detection', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./options/LOLv2-Real.yml")
    args = parser.parse_args()
    opt = parse(args.config)
    main(opt)
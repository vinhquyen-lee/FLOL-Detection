import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import cv2
import numpy as np
import time
from model.flol import create_model
from options.options import parse
from ultralytics import YOLO  # <--- [1] [THÊM] Import thư viện YOLO

def pad_tensor(tensor, multiple=8):
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor

# [SỬA] Thêm tham số yolo_path vào hàm main
def main(opt, input_path, output_path, scale_percent, yolo_path):
    # 1. Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang chạy trên thiết bị: {device} ---")

    # 2. Load Model FLOL
    print("⏳ Đang tải mô hình FLOL (Làm sáng)...")
    model = create_model()
    weights_path = opt['settings']['weight']
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['params'])
    model.to(device)
    model.eval()
    print("✅ Đã tải FLOL thành công!")

    # --- [2] [THÊM] LOAD MODEL YOLO ---
    print(f"⏳ Đang tải mô hình YOLO từ: {yolo_path} ...")
    try:
        yolo_model = YOLO(yolo_path)
        print("✅ Đã tải YOLO thành công!")
    except Exception as e:
        print(f"❌ Lỗi tải YOLO: {e}")
        return
    # ----------------------------------

    # 3. Mở Video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Lỗi: Không mở được video {input_path}")
        return

    # Lấy thông số gốc
    org_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    org_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if scale_percent < 100:
        new_width = int(org_width * scale_percent / 100)
        new_height = int(org_height * scale_percent / 100)
        print(f"Đang RESIZE video: {org_width}x{org_height} -> {new_width}x{new_height}")
    else:
        new_width = org_width
        new_height = org_height
        print(f"Giữ nguyên độ phân giải: {org_width}x{org_height}")

    # 4. Video Writer 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    to_tensor = transforms.ToTensor()
    frame_count = 0
    start_time = time.time()

    print("🚀 Bắt đầu xử lý Combo FLOL + YOLO... (Nhấn 'q' để dừng sớm)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- RESIZE ---
        if scale_percent < 100:
            frame_processing = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            frame_processing = frame

        # --- XỬ LÝ FLOL (LÀM SÁNG) ---
        img_rgb = cv2.cvtColor(frame_processing, cv2.COLOR_BGR2RGB)
        img_tensor = to_tensor(img_rgb).unsqueeze(0).to(device)
        
        # Padding 
        _, _, H, W = img_tensor.shape
        img_padded = pad_tensor(img_tensor)

        with torch.no_grad():
            output = model(img_padded)

        # Hậu xử lý FLOL -> Ra ảnh sáng (output_bgr)
        output = torch.clamp(output, 0., 1.)
        output = output[:, :, :H, :W] # Cắt bỏ phần padding
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        output_bgr = (output_np * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR)

        # --- [3] [THÊM] CHẠY YOLO NHẬN DIỆN ---
        # Lấy ảnh đã làm sáng (output_bgr) đưa vào YOLO
        # conf=0.4: Chỉ hiện khung nếu độ tin cậy > 40%
        results = yolo_model(output_bgr, verbose=False, conf=0.4)
        
        # Lấy ảnh đã được vẽ khung nhận diện (Annotated Frame)
        final_frame = results[0].plot()
        # ---------------------------------------

        # [SỬA] Ghi ảnh cuối cùng (đã có khung) vào video
        out.write(final_frame)

        frame_count += 1
        if frame_count % 10 == 0: 
            elapsed = time.time() - start_time
            process_fps = frame_count / elapsed
            print(f"\rTiến độ: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) | Tốc độ: {process_fps:.1f} FPS", end="")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n\n✅ Xong! Video đã lưu tại: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./options/LOLv2-Real.yml")
    
    parser.add_argument("--input", type=str, 
                        default="datasets/LOLv2-Real/test/Low/test6.mp4", 
                        help="Đường dẫn file video đầu vào")

    parser.add_argument("--output", type=str, 
                        default="results/LOLv2-Real/test6_result.mp4", 
                        help="Đường dẫn file video kết quả")

    parser.add_argument("--scale", type=int, default=50, help="Tỷ lệ % resize")

    # [THÊM] Tham số đường dẫn file best.pt
    parser.add_argument("--yolo", type=str, default="yolo11n.pt", help="Đường dẫn file trọng số YOLO")
    
    args = parser.parse_args()
    opt = parse(args.config)

    import os
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tự động tạo thư mục: {output_dir}")

    # [SỬA] Truyền thêm tham số args.yolo
    main(opt, args.input, args.output, args.scale, args.yolo)
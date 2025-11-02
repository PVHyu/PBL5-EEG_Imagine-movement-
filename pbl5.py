import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt

# =========================================================
# Bước 1: Khởi tạo & load datasheet
# =========================================================
gdf_path = "D:/UNIVERSITY/PBL5/Datasets_PBL5/A01T.gdf"

# Đọc dữ liệu GDF
raw = mne.io.read_raw_gdf(gdf_path, preload=True)
print("Đã load file EEG:", gdf_path)

# =========================================================
# Bước 2: Cấu hình tham số
# =========================================================
fs = raw.info['sfreq']            # Tần số lấy mẫu
window_len = 2.5                  # Thời gian mỗi epoch (giây)
offset = 0.5                      # Thời gian trễ (giây)
n_channels = 22                   # Số kênh EEG sử dụng

# Lấy dữ liệu EEG gốc (22 kênh đầu)
data_raw = raw.get_data()[:n_channels, :]

# Thiết kế bộ lọc FIR band-pass 8–30 Hz
bp_coeff = firwin(101, [8, 30], pass_zero=False, fs=fs)
print("Đã thiết kế bộ lọc FIR 8–30 Hz")

# =========================================================
# Bước 3: Tách epoch & tiền xử lý
# =========================================================
# Lấy thông tin cue onset từ annotations
events, event_id = mne.events_from_annotations(raw)
print("Danh sách sự kiện:", event_id)
print("Tổng số sự kiện:", len(events))

samples_window = int(window_len * fs)
samples_offset = int(offset * fs)

epochs = []
labels = []

for event in events:
    onset = event[0]              # vị trí mẫu của cue onset
    label = event[2]              # mã sự kiện (class)
    
    start = int(onset + samples_offset)
    stop = int(start + samples_window)
    
    # Kiểm tra nằm trong giới hạn dữ liệu
    if stop <= data_raw.shape[1]:
        # Cắt epoch
        epoch = data_raw[:, start:stop]
        
        # Lọc tín hiệu từng epoch
        epoch_filt = filtfilt(bp_coeff, [1.0], epoch, axis=1)
        
        # Zero-mean theo kênh
        epoch_zm = epoch_filt - np.mean(epoch_filt, axis=1, keepdims=True)
        
        # Lưu lại
        epochs.append(epoch_zm)
        labels.append(label)

epochs = np.array(epochs)
labels = np.array(labels)

print(f"Đã tách {len(epochs)} epoch | Mỗi epoch dài {epochs.shape[2]/fs:.2f}s ({epochs.shape[2]} mẫu).")

# # =========================================================
# # Hiển thị dạng sóng của một epoch bất kỳ
# epoch_idx = 10  # ví dụ: epoch thứ 10
# plt.figure(figsize=(10, 4))
# plt.plot(epochs[epoch_idx].T)
# plt.title(f"Dạng sóng EEG - Epoch {epoch_idx+1} | Label: {labels[epoch_idx]}")
# plt.xlabel("Thời gian (mẫu)")
# plt.ylabel("Biên độ (µV)")
# plt.show()

# =========================================================
# Bước 4: Dọn dữ liệu
# =========================================================

# Kiểm tra và loại epoch chứa NaN
valid_idx = [i for i in range(len(epochs)) if not np.isnan(epochs[i]).any()]
epochs = epochs[valid_idx]
labels = labels[valid_idx]

print(f"Đã loại bỏ epoch chứa NaN. Còn lại {len(epochs)} epoch hợp lệ.")

# Kiểm tra số lượng trial theo từng lớp
unique, counts = np.unique(labels, return_counts=True)
print("Phân bố số trial theo lớp:")
for u, c in zip(unique, counts):
    print(f"  Lớp {u}: {c} trials")

# =========================================================
# Bước 5: Lọc chỉ giữ các lớp MI
# =========================================================

# Giữ lại chỉ các lớp MI (Motor Imagery): 7, 8, 9, 10
valid_classes = [7, 8, 9, 10]
sel_idx = np.isin(labels, valid_classes)
epochs = epochs[sel_idx]
labels = labels[sel_idx]

print(f"Sau khi lọc MI, còn {len(epochs)} epoch")
print("Phân bố lớp MI:")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Lớp {u}: {c} trials")

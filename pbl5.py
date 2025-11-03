# =========================================================
# TOÀN BỘ PIPELINE: A01T.gdf → CSP TRAIN + LƯU THAM SỐ (ĐÃ SỬA 100%)
# =========================================================
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt
from scipy.linalg import eigh
import pickle

# --------------------- BƯỚC 1: LOAD DATA ---------------------
gdf_path = "D:/UNIVERSITY/PBL5/Datasets_PBL5/A01T.gdf"
print("Đang load file EEG:", gdf_path)
raw = mne.io.read_raw_gdf(gdf_path, preload=True)

fs = raw.info['sfreq']
print(f"Sampling frequency: {fs} Hz")

# --------------------- BƯỚC 2: CẤU HÌNH ---------------------
window_len = 2.5
offset = 0.5
n_channels = 22
samples_window = int(window_len * fs)
samples_offset = int(offset * fs)

data_raw = raw.get_data()[:n_channels, :]
bp_coeff = firwin(101, [8, 30], pass_zero=False, fs=fs)
print("Đã thiết kế bộ lọc FIR 8–30 Hz")

# --------------------- BƯỚC 3: TÁCH EPOCH ---------------------
events, event_id = mne.events_from_annotations(raw)
print("Danh sách sự kiện:", event_id)

epochs = []
labels = []

for event in events:
    onset = event[0]
    label = event[2]
    
    start = int(onset + samples_offset)
    stop = int(start + samples_window)
    
    if stop <= data_raw.shape[1]:
        epoch = data_raw[:, start:stop]
        epoch_filt = filtfilt(bp_coeff, [1.0], epoch, axis=1)
        epoch_zm = epoch_filt - np.mean(epoch_filt, axis=1, keepdims=True)
        epochs.append(epoch_zm)
        labels.append(label)

epochs = np.array(epochs)
labels = np.array(labels)
print(f"Đã tách {len(epochs)} epoch | Mỗi epoch: {epochs.shape[2]} mẫu")

# --------------------- BƯỚC 4: DỌN DỮ LIỆU ---------------------
valid_idx = [i for i in range(len(epochs)) if not np.isnan(epochs[i]).any()]
epochs = epochs[valid_idx]
labels = labels[valid_idx]
print(f"Sau loại NaN: {len(epochs)} epoch")

# --------------------- BƯỚC 5: LỌC MI (7,8,9,10) ---------------------
valid_classes = [7, 8, 9, 10]
sel_idx = np.isin(labels, valid_classes)
epochs = epochs[sel_idx]
labels = labels[sel_idx]

print(f"Sau lọc MI: {len(epochs)} epoch")
print("Phân bố lớp MI:")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Lớp {u}: {c} trials")

label_map = {7: 0, 8: 1, 9: 2, 10: 3}
y = np.array([label_map[l] for l in labels])

# --------------------- BƯỚC 6: CSP ONE-VS-REST ---------------------
X = epochs
n_trials, n_channels, n_times = X.shape
n_classes = 4
m = 2

def compute_csp_ovr(X, y, m=2):
    W_list = []
    for c in range(n_classes):
        X_c = X[y == c]
        X_rest = X[y != c]
        cov_c = np.mean([x @ x.T for x in X_c], axis=0)
        cov_rest = np.mean([x @ x.T for x in X_rest], axis=0)
        cov_c /= np.trace(cov_c)
        cov_rest /= np.trace(cov_rest)
        eigenvals, eigenvecs = eigh(cov_c, cov_rest)
        idx = np.argsort(eigenvals)[::-1]
        W = eigenvecs[:, idx]
        W_csp = np.hstack([W[:, :m], W[:, -m:]])
        W_list.append(W_csp)
    return W_list

print(f"\nĐang tính CSP (One-vs-Rest, m={m}) trên {n_trials} trial...")
W_csp_list = compute_csp_ovr(X, y, m=m)
print(f"Hoàn tất! {len(W_csp_list)} bộ lọc CSP (mỗi class: {2*m} filter)")

# --------------------- TRÍCH ĐẶC TRƯNG LOG-VARIANCE (ĐÃ SỬA) ---------------------
def extract_csp_features(X, W_list, m=2):
    n_trials = X.shape[0]  # ĐÃ SỬA: n_trials, KHÔNG PHẢI n_standard
    n_filters_per_class = 2 * m
    n_total_features = len(W_list) * n_filters_per_class
    features = np.zeros((n_trials, n_total_features))
    
    for i in range(n_trials):
        trial = X[i]  # (n_channels, n_times)
        feat_idx = 0
        for W in W_list:
            X_proj = W.T @ trial  # (2m, n_times)
            var = np.var(X_proj, axis=1)
            logvar = np.log(var + 1e-12)
            features[i, feat_idx:feat_idx + n_filters_per_class] = logvar
            feat_idx += n_filters_per_class
    return features

print("Đang trích đặc trưng CSP...")
X_csp_features = extract_csp_features(X, W_csp_list, m)
print(f"Đặc trưng CSP: {X_csp_features.shape} → {n_trials} trials × {n_classes * 2 * m} dims")

# --------------------- HIỂN THỊ KẾT QUẢ ---------------------
# 1. 2D Feature Space
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'purple']
for c in range(n_classes):
    idx = (y == c)
    plt.scatter(X_csp_features[idx, 0], X_csp_features[idx, 4], 
                c=colors[c], label=f'Class {c}', alpha=0.7, s=60)
plt.xlabel("log-var (Class 0 - Filter 1)")
plt.ylabel("log-var (Class 1 - Filter 1)")
plt.title("CSP Feature Space – 2D Projection")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. In mẫu
print("\n=== MẪU ĐẶC TRƯNG CSP (5 trial đầu) ===")
for i in range(5):
    print(f"Trial {i:2d} | Label: {y[i]} | Features[:8]: {X_csp_features[i, :8].round(4)}")

# --------------------- LƯU THAM SỐ ---------------------
csp_params = {
    'W_csp_list': W_csp_list,
    'label_map': label_map,
    'm': m,
    'n_channels': n_channels,
    'fs': fs,
    'window_len': window_len,
    'offset': offset,
    'bp_coeff': bp_coeff,
    'X_csp_mean': X_csp_features.mean(axis=0),
    'X_csp_std': X_csp_features.std(axis=0) + 1e-8
}

save_path = "csp_params_A01T_train.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(csp_params, f)

print(f"\nHOÀN TẤT! ĐÃ LƯU THAM SỐ → {save_path}")
print("   Sẵn sàng dùng cho A01E.gdf!")
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, firwin

# Đường dẫn dữ liệu EEG
gdf_path = "D:/UNIVERSITY/PBL5/Datasets_PBL5/A01T.gdf"

# 1) Đọc dữ liệu GDF
raw = mne.io.read_raw_gdf(gdf_path, preload=True)
fs = raw.info['sfreq']
data_raw = raw.get_data()[:22, :]

# 2) Thiết kế và áp dụng bộ lọc FIR 8–30Hz
bp_coeff = firwin(101, [8, 30], pass_zero=False, fs=fs)
data_filt = filtfilt(bp_coeff, [1.0], data_raw, axis=1)

# 3) Zero-mean theo kênh
data_zm = data_filt - np.mean(data_filt, axis=1, keepdims=True)

# 4) Hiển thị 5 giây đầu của kênh 1
t = np.arange(data_raw.shape[1]) / fs
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:int(fs*5)], data_raw[0, :int(fs*5)], color='gray')
plt.title("EEG gốc - Trước xử lý (Kênh 1)")
plt.subplot(2, 1, 2)
plt.plot(t[:int(fs*5)], data_zm[0, :int(fs*5)], color='blue')
plt.title("EEG sau lọc 8–30Hz & Zero-mean (Kênh 1)")
plt.xlabel("Thời gian (s)")
plt.tight_layout()
plt.show()

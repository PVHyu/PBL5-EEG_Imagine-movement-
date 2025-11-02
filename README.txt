Các bước huấn luyện mô hình nhận diện tín hiệu sóng não
1. Huấn luyện trên máy tính

Bước 1: Khởi tạo & load datasheet

Bước 2: Cấu hình tham số: 
        Lấy fs, đặt window_len = 2.5s, offset = 0.5s, 
        chọn 22 kênh EEG, thiết kế FIR band-pass 8–30 Hz.

Bước 3: Tách epoch & tiền xử lý trên từng epoch: 
        Dò cue onset, cắt epoch (start = latency + offset), 
        lọc bằng filtfilt, zero-mean theo kênh, gán nhãn (class1..4).

Bước 4: Dọn dữ liệu: 
        Loại những epoch bị đánh dấu Rejection, 
        loại NaN, kiểm tra số lượng trial.

Bước 5: Chia hold-out: 
        Test = trial 1..58, Train = trial 59..N; in phân bố lớp.

Bước 6: Trích đặc trưng (CSP One-vs-Rest)
        Với mỗi class: tính ma trận hiệp phương sai chuẩn hoá, 
        giải eigen tổng quát, lấy m thành phần lớn nhất & nhỏ nhất 
        → tính log-variance làm feature.

Bước 7: Chuẩn hoá & huấn luyện (rLDA):
        Z-score theo train (mu_feat, std_feat), 
        huấn luyện rLDA (shrinkage Gamma), xuất Wlda và b cho suy luận.

Bước 8: Áp dụng lên test, đánh giá & lưu tham số
        Trích feature test bằng CSP từ train, chuẩn hoá, 
        dự đoán (score = X * Wlda + b), in accuracy + confusion matrix, 
        lưu tham số (params_holdout.mat) để triển khai.

👉 Phần này giống như “học trước” để FPGA chỉ cần “áp dụng” chứ không phải tự học.

2. Thực thi trên FPGA

Bước 1: FPGA nhận tín hiệu EEG từ ADC.

Bước 2: Chạy bộ lọc băng thông 8–30 Hz (chỉ giữ sóng cần thiết).

Bước 3: Áp dụng ma trận CSP (đã huấn luyện) để trộn các kênh EEG → ra vài kênh mới.

Bước 4: Trong mỗi cửa sổ 1 giây, tính “mức năng lượng” từng kênh → rồi lấy log.

Bước 5: Đưa các đặc trưng đó vào công thức LDA (w·f + b) → cho ra kết quả lớp (trái/phải).

Bước 6: FPGA biến kết quả phân loại thành lệnh điều khiển xe lăn (ví dụ tiến, lùi, rẽ).
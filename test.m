% Đọc dữ liệu GDF mà không cần EEGLAB
[data, header] = sload('D:/UNIVERSITY/PBL5/Datasets_PBL5/A01E.gdf');

fs = header.SampleRate;
t = (0:length(data)-1) / fs;

% Chọn kênh để hiển thị, ví dụ kênh 1
raw_signal = data(:,1);

% Hiển thị trước khi lọc
figure;
subplot(2,1,1);
plot(t, raw_signal);
title('Tín hiệu gốc');
xlabel('Thời gian (s)');
ylabel('Biên độ');
grid on;

% Thiết kế và lọc 8–30 Hz
bpFilt = designfilt('bandpassfir', ...
    'FilterOrder', 100, ...
    'CutoffFrequency1', 8, ...
    'CutoffFrequency2', 30, ...
    'SampleRate', fs);

filtered_signal = filtfilt(bpFilt, raw_signal);

% Hiển thị sau khi lọc
subplot(2,1,2);
plot(t, filtered_signal);
title('Tín hiệu sau khi lọc 8–30 Hz');
xlabel('Thời gian (s)');
ylabel('Biên độ');
grid on;

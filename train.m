%% ======================= HOLD-OUT (A01T, 22 EEG channels) =======================
clear all; clc; close all;
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% === Load GDF ===
EEG = pop_biosig('A01T.gdf'); 
EEG = eeg_checkset(EEG);

fs = EEG.srate;
window_len = 2.5 * fs;    % 2.5s cửa sổ
offset     = 0.5 * fs;    % 0.5s sau cue

% === 22 kênh EEG (bỏ 3 kênh EOG) ===
data_raw = double(EEG.data(1:22,:));
numCh = size(data_raw,1);

% === FIR Band-pass 8–30 Hz ===
bpFilt = designfilt('bandpassfir','FilterOrder',100, ...
    'CutoffFrequency1',8,'CutoffFrequency2',30,'SampleRate',fs);

%% 1) Trích toàn bộ trial có nhãn
X = []; Y = [];
for i = 1:length(EEG.event)
    e = EEG.event(i);
    if ischar(e.type) && contains(e.type,'cue onset')
        % Bỏ trial bị reject ngay sau cue (nếu có)
        if i < length(EEG.event)
            next = EEG.event(i+1).type;
            if ischar(next) && contains(next,'Rejection')
                continue;
            end
        end
        idx_start = round(e.latency + offset);
        idx_end   = idx_start + window_len - 1;
        if idx_end <= size(data_raw,2)
            seg    = data_raw(:,idx_start:idx_end);
            seg_f  = filtfilt(bpFilt, seg')';     % lọc băng
            seg_zm = seg_f - mean(seg_f,2);       % zero-mean theo kênh

            X(end+1,:,:) = seg_zm; %#ok<SAGROW>
            if     contains(e.type,'class1'), Y(end+1)=1;
            elseif contains(e.type,'class2'), Y(end+1)=2;
            elseif contains(e.type,'class3'), Y(end+1)=3;
            elseif contains(e.type,'class4'), Y(end+1)=4;
            else,  Y(end+1)=NaN; % phòng hờ
            end
        end
    end
end
X = X(~isnan(Y),:,:); 
Y = Y(~isnan(Y));
nAll = size(X,1);

fprintf("✅ Epoch xong: %d trial có nhãn.\n", nAll);
fprintf("📊 Biên độ sau epoch: min = %.6f, max = %.6f\n", min(X(:)), max(X(:)));
assert(size(X,1)==length(Y), 'Mismatch X và Y!');

%% 2) Chia hold-out (Test: 1..58, Train: 59..N)
nTest = 58;
if nAll < (nTest + 1)
    error("Không đủ trial để tách Test=1..58 và Train=59..N. Đang có %d trial.", nAll);
end
idxTest  = 1:nTest;                  % 1..58
idxTrain = (nTest+1):nAll;           % 59..N (kỳ vọng N=288)

X_test  = X(idxTest,:,:);   Y_test  = Y(idxTest);
X_train = X(idxTrain,:,:);  Y_train = Y(idxTrain);

fprintf("🔧 Split: Test=%d (1..58), Train=%d (59..%d)\n", ...
    length(Y_test), length(Y_train), nAll);

% (tuỳ chọn) In phân bố lớp
numClasses = 4;
fprintf("📊 Phân bố lớp (Test 1..58):\n");
for c = 1:numClasses
    fprintf("  - Class %d: %d\n", c, sum(Y_test==c));
end
fprintf("📊 Phân bố lớp (Train 59..%d):\n", nAll);
for c = 1:numClasses
    fprintf("  - Class %d: %d\n", c, sum(Y_train==c));
end

%% 3) Train pipeline trên TRAIN (CSP OVR + chuẩn hoá + rLDA)
m = 2;                                % 2 cặp CSP mỗi class (→ 4 feat/class)
cov_norm = @(trial) (trial*trial')/trace(trial*trial');

% --- CSP OVR ---
features_train = []; 
Wcsp_all = cell(1,numClasses);
for c = 1:numClasses
    Xc = X_train(Y_train==c,:,:);
    Xr = X_train(Y_train~=c,:,:);
    if isempty(Xc) || isempty(Xr)
        error("TRAIN: lớp %d không có dữ liệu!", c);
    end

    % Trung bình hiệp phương sai đã chuẩn hoá theo vết
    Cc = zeros(numCh); Cr = zeros(numCh);
    for i = 1:size(Xc,1), Cc = Cc + cov_norm(squeeze(Xc(i,:,:))); end
    for i = 1:size(Xr,1), Cr = Cr + cov_norm(squeeze(Xr(i,:,:))); end
    Cc = Cc/size(Xc,1); Cr = Cr/size(Xr,1);

    % Giải tổng quát (regularize nhỏ để ổn định)
    [EVec,EVal] = eig(Cc, Cc+Cr+1e-9*eye(numCh));
    [~,ind] = sort(diag(EVal),'descend'); 
    W = EVec(:,ind);

    % Lấy m thành phần lớn nhất & m thành phần nhỏ nhất
    Wcsp = [W(:,1:m), W(:,end-m+1:end)];
    Wcsp_all{c} = Wcsp;

    % Đặc trưng log-variance (chuẩn hoá tổng phương sai)
    feat_c = zeros(size(X_train,1), 2*m);
    for i = 1:size(X_train,1)
        Z = Wcsp' * squeeze(X_train(i,:,:));
        v = var(Z,0,2); 
        feat_c(i,:) = log(v/sum(v));
    end
    features_train = [features_train feat_c]; %#ok<AGROW>
end

% --- Z-score theo TRAIN ---
mu_feat  = mean(features_train,1);
std_feat = std(features_train,[],1);
std_feat(std_feat==0) = 1;     % tránh chia 0
Xfeat_train = (features_train - mu_feat) ./ std_feat;

% --- rLDA (shrinkage) ---
Gamma = 0.0026; 
Delta = 0; 
Mdl = fitcdiscr(Xfeat_train, Y_train, ...
    'DiscrimType','linear', 'Gamma', Gamma, 'Delta', Delta);

% --- Xuất (W,b) tương đương cho suy luận nhanh ---
classes = unique(Y_train); 
Kc = numel(classes); 
D  = size(Xfeat_train,2);

mu = zeros(D,Kc); priors = zeros(Kc,1);
for kclass = 1:Kc
    mu(:,kclass)  = mean(Xfeat_train(Y_train==classes(kclass),:),1)';
    priors(kclass)= mean(Y_train==classes(kclass));
end
Sigma = cov(Xfeat_train);
Sigma_shrunk = (1-Gamma)*Sigma + Gamma*diag(diag(Sigma));
Sigma_shrunk = Sigma_shrunk + max(1e-6,Delta)*eye(D);
invSigma = inv(Sigma_shrunk);

Wlda = zeros(D,Kc); 
b    = zeros(Kc,1);
for kclass = 1:Kc
    Wlda(:,kclass) = invSigma * mu(:,kclass);
    b(kclass) = -0.5*(mu(:,kclass)'*invSigma*mu(:,kclass)) + log(priors(kclass)+eps);
end

%% 4) Áp dụng hệ số TRAIN cho TEST và đánh giá
% --- Trích đặc trưng TEST bằng Wcsp_all từ TRAIN ---
features_test = [];
for c = 1:numel(classes)
    Wcsp = Wcsp_all{c};
    feat_c = zeros(size(X_test,1), 2*m);
    for i = 1:size(X_test,1)
        Z = Wcsp' * squeeze(X_test(i,:,:));
        v = var(Z,0,2);
        feat_c(i,:) = log(v/sum(v));
    end
    features_test = [features_test feat_c]; %#ok<AGROW>
end

% --- Z-score TEST theo (mu,std) của TRAIN ---
Xfeat_test = (features_test - mu_feat) ./ std_feat;

% --- Phân loại ---
scores = Xfeat_test * Wlda + repmat(b', size(Xfeat_test,1), 1);
[~, idx_pred] = max(scores, [], 2);
Y_pred = classes(idx_pred);

% --- Độ chính xác & ma trận nhầm lẫn ---
acc = mean(Y_pred(:)==Y_test(:)) * 100;
fprintf("\n🎯 HOLD-OUT Accuracy (Test 1..58) = %.2f%%\n", acc);

C = confusionmat(Y_test(:), Y_pred(:), 'Order', 1:numClasses);
disp('📌 Confusion matrix (rows: true, cols: pred):'); 
disp(C);

per_class_acc = 100*diag(C)./max(1,sum(C,2));
for c = 1:numClasses
    fprintf("  - Class %d acc: %.2f%%\n", c, per_class_acc(c));
end

%% In range hệ số đã train
fprintf("\n📊 Range hệ số (TRAIN):\n");
coeffs = bpFilt.Coefficients;
fprintf("FIR coeffs:   min = %.6f, max = %.6f\n", min(coeffs), max(coeffs));

W_all = cell2mat(Wcsp_all');  % ghép tất cả CSP matrix
fprintf("CSP filters:  min = %.6f, max = %.6f\n", min(W_all(:)), max(W_all(:)));
fprintf("LDA Wlda:     min = %.6f, max = %.6f\n", min(Wlda(:)), max(Wlda(:)));
fprintf("LDA bias b:   min = %.6f, max = %.6f\n", min(b(:)), max(b(:)));
fprintf("mu_feat:      min = %.6f, max = %.6f\n", min(mu_feat(:)), max(mu_feat(:)));
fprintf("std_feat:     min = %.6f, max = %.6f\n", min(std_feat(:)), max(std_feat(:)));
inv_std_feat = 1 ./ std_feat;
fprintf("1/std_feat:   min = %.6f, max = %.6f\n", min(inv_std_feat(:)), max(inv_std_feat(:)));

%% Lưu hệ số (từ TRAIN) để lượng tử/nạp FPGA
save("params_holdout.mat", "bpFilt","Wcsp_all","Wlda","b","mu_feat","std_feat","inv_std_feat");
fprintf("💾 Đã lưu tham số hold-out vào params_holdout.mat (hệ số học từ TRAIN 59..%d)\n", nAll);

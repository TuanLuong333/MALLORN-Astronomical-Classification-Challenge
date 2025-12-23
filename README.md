# MALLORN Astronomical Classification Challenge

## *Điểm Public Leaderboard: 0.6649*

Repository này trình bày lời giải đạt 0.6649 cho bài toán **MALLORN Astronomical Classification Challenge**. Trọng tâm là pipeline LightGBM với feature engineering sâu trên lightcurve đa băng lọc, hiệu chỉnh suy giảm ánh sáng (EBV), bổ sung đặc trưng vật lý (rest-frame), và tối ưu ngưỡng theo đường cong Precision-Recall để tối đa F1.

# 1. Bài toán
Mục tiêu là xây dựng mô hình phân loại **Tidal Disruption Events (TDEs)** từ dữ liệu lightcurve giả lập của LSST, dựa trên đo sáng theo thời gian của 6 bộ lọc (`u, g, r, i, z, y`) cùng với metadata (redshift, extinction, ...).

## Dữ liệu
Dữ liệu gồm 2 phần chính:
* **Log files (`train_log.csv`, `test_log.csv`)**: chứa `object_id`, `Z`, `Z_err`, `EBV`, `SpecType`, `target`, và thông tin `split`.
* **Lightcurve files**: nằm trong các thư mục `split_01` ... `split_20`, mỗi file `[train/test]_full_lightcurves.csv` chứa chuỗi thời gian theo từng bộ lọc.

## Thách thức
* **Mất cân bằng lớp**: TDE cực hiếm so với các loại transient khác.
* **Chuỗi thời gian thưa và không đều**: số điểm quan sát theo thời gian khác nhau cho mỗi object và mỗi filter.
* **Chỉ số đánh giá**: **F1 Score**, nhấn mạnh cân bằng giữa Precision và Recall.

# System Design
Pipeline theo hướng “feature engineering trước, mô hình tabular sau”, tập trung vào xử lý vật lý (de-extinction), đặc trưng thống kê theo filter, sau đó tổng hợp mức object và huấn luyện LightGBM với threshold tối ưu.

### Tầng dữ liệu
- **Metadata** (`train_log.csv`, `test_log.csv`): dùng để ghép `Z`, `EBV`, `Z_err`, `split`...
- **Lightcurves**: đọc theo từng split, gom về dạng long-format cho feature extraction.

### Chiến lược Feature Engineering
Trọng tâm là tạo đặc trưng “giàu thông tin” từ lightcurve thưa:
- **Per-filter stats**: thống kê cơ bản (mean/std/median/min/max), quantile, skew/kurtosis, SNR, độ biến thiên, khoảng thời gian quan sát.
- **Đặc trưng động học**: độ dốc rise/decay, vị trí peak, độ rộng đỉnh (width 50/75%), cadence, sign-change.
- **Tích phân diện tích**: trapezoid area, cân bằng dương/âm, energy của flux/SNR.
- **Detection features**: số điểm có `SNR >= 3` và flux dương, span của vùng phát hiện.

### Tổng hợp toàn bộ filter
Sau khi có đặc trưng theo từng filter, tiếp tục tạo:
- **Global features** theo object (gộp tất cả filter).
- **Categorical features**: `first_filter`, `peak_filter`.

### Bổ sung vật lý thiên văn
Từ `Z` và `EBV`:
- **De-extinction** theo mô hình CCM89.
- **Luminosity distance** và **distance modulus**.
- **Rest-frame features**: hiệu chỉnh thời gian/slope theo `(1+z)`.
- **Đặc trưng độ sáng tuyệt đối** từ peak flux.

### Mô hình & vòng lặp huấn luyện
Sử dụng **LightGBM** (tabular, xử lý missing tốt), kèm **scale_pos_weight** để cân bằng lớp, và **threshold optimization** theo PR-curve để tối đa F1.

# 2. Tổng quan hướng tiếp cận
1. **Nạp dữ liệu & tự phát hiện DATA_DIR** từ môi trường Kaggle.
2. **De-extinction** flux theo EBV (CCM89) trước khi trích xuất đặc trưng.
3. **Feature engineering sâu** theo từng filter + tổng hợp theo object.
4. **Bổ sung đặc trưng vật lý** (rest-frame, luminosity distance, magnitude).
5. **Huấn luyện LightGBM** với 5-fold Stratified CV và tối ưu ngưỡng dựa trên PR-curve.
6. **Tùy chọn**: phân tích tương quan Spearman để loại bớt đặc trưng dư thừa.

# 3. Kỹ thuật Machine Learning
## Mô hình chính
**LightGBM Classifier** với cấu hình:
* `n_estimators=6500`, `learning_rate=0.02`, `num_leaves=127`
* `subsample=0.8`, `colsample_bytree=0.75`, `feature_fraction_bynode=0.8`
* `reg_alpha=0.4`, `reg_lambda=6.0`, `min_child_samples=20`
* `extra_trees=True`, `scale_pos_weight=neg/pos`

## Chuẩn bị dữ liệu
* Điền median cho numeric.
* Thêm nhãn `__MISSING__` cho categorical.
* Loại cột hằng số.

## Tối ưu ngưỡng (threshold)
Mỗi fold tìm `threshold` tốt nhất theo **PR-curve** để tối đa F1, sau đó chọn ngưỡng toàn cục từ OOF.

# 4. Phân tích tương quan & Feature Pruning (tùy chọn)
Notebook có cell đánh giá tương quan Spearman để:
* Xác định cặp đặc trưng tương quan cao.
* Xếp hạng bằng importance của LightGBM.
* Loại đặc trưng dư thừa nếu `corr > 0.98`.

# 5. Pipeline chạy cuối
1. Load `train_log.csv`, `test_log.csv` và lightcurves theo split.
2. Tạo bảng hiệu chỉnh EBV và de-extinction flux.
3. Trích xuất đặc trưng per-filter + global.
4. Bổ sung đặc trưng vật lý theo redshift.
5. Chuẩn hóa dữ liệu, xử lý missing, xử lý categorical.
6. 5-fold CV LightGBM + tìm threshold tối ưu.
7. Dự đoán `test_prob` và tạo `submission.csv`.

# 6. Thông tin kỹ thuật
* **Thư viện**: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `matplotlib`, `seaborn`.
* **Validation**: 5-fold Stratified CV, metric F1.
* **Xử lý mất cân bằng**: `scale_pos_weight`.
* **Notebook chính**: `mallorn_0.6649.ipynb`.


from __future__ import annotations

import numpy as np


def harrell_c_index(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    """Compute Harrell's concordance index.

    times: survival times
    events: 1 if event observed, 0 if censored
    scores: risk scores (higher = more risk)
    """
    num = 0.0
    den = 0.0
    n = len(times)
    for i in range(n):
        for j in range(n):
            if times[i] < times[j] and events[i] == 1:
                den += 1
                if scores[i] > scores[j]:
                    num += 1
                elif scores[i] == scores[j]:
                    num += 0.5
    return num / den if den > 0 else 0.0

def bootstrap_c_index(times: np.ndarray, events: np.ndarray, scores: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """
    Tính P-Value và 95% CI cho C-Index bằng phương pháp Bootstrapping.
    """
    rng = np.random.default_rng(seed)
    n = len(times)
    c_indices = []

    print(f"[INFO] Đang chạy Bootstrapping ({n_bootstraps} lần) để tính 95% CI & P-Value...")
    for _ in range(n_bootstraps):
        # Lấy mẫu ngẫu nhiên có hoàn lại (sampling with replacement)
        indices = rng.choice(n, n, replace=True)
        t_b = times[indices]
        e_b = events[indices]
        s_b = scores[indices]

        # Bỏ qua các batch không có event nào để tránh lỗi chia cho 0
        if np.sum(e_b) > 0:
            c_idx = harrell_c_index(t_b, e_b, s_b)
            if c_idx > 0:
                c_indices.append(c_idx)

    c_indices = np.array(c_indices)
    if len(c_indices) == 0:
        return float('nan'), float('nan'), float('nan')

    # 1. 95% Confidence Interval (Khoảng tin cậy 95%)
    lower_ci = float(np.percentile(c_indices, 2.5))
    upper_ci = float(np.percentile(c_indices, 97.5))

    # 2. P-Value: Kiểm định giả thuyết H0 (Mô hình không tốt hơn đoán ngẫu nhiên, tức C-Index <= 0.5)
    p_value = float(np.mean(c_indices <= 0.5))

    return p_value, lower_ci, upper_ci

def calculate_iauc(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    """
    Tính Integrated AUC (Time-dependent AUC) sử dụng scikit-survival.
    """
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        
        # Định dạng mảng cấu trúc mà sksurv yêu cầu
        y = np.empty(len(times), dtype=[('event', bool), ('time', float)])
        y['event'] = events.astype(bool)
        y['time'] = times

        # Tìm khoảng thời gian hợp lệ để tính toán
        event_mask = events == 1
        if np.sum(event_mask) < 2:
            return float('nan')
            
        t_min = times[event_mask].min()
        t_max = times[event_mask].max()

        # Chia đều 100 mốc thời gian từ đầu đến cuối
        time_points = np.linspace(t_min + 1, t_max - 1, 100)

        # Trả về Mean I-AUC
        _, mean_auc = cumulative_dynamic_auc(y, y, scores, time_points)
        return float(mean_auc)
        
    except ImportError:
        print("[WARN] Thư viện 'scikit-survival' không tồn tại. Tính năng I-AUC sẽ trả về NA.")
        print("       >>> Hướng dẫn: pip install scikit-survival")
        return float('nan')
    except Exception as e:
        print(f"[WARN] Lỗi tính I-AUC: {e}")
        return float('nan')
    
def calculate_cv_stats(metrics_list: list[float]) -> tuple[float, float, float, float, float, int]:
    """
    Tính Mean, STD, 95% CI, Max, và Argmax từ danh sách các chỉ số (ví dụ: 5 C-Index).
    Trả về: (Mean, STD, CI_lower, CI_upper, Max, Argmax)
    """
    arr = np.array(metrics_list)
    arr = arr[~np.isnan(arr)] # Lọc bỏ các giá trị lỗi
    if len(arr) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), -1
        
    mean_val = float(np.mean(arr))
    # ddof=1 để tính độ lệch chuẩn mẫu (Sample Standard Deviation)
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    max_val = float(np.max(arr))
    
    # Lấy index của fold tốt nhất từ list gốc
    max_idx = int(np.argmax(np.array(metrics_list))) 

    # 95% Confidence Interval của Trung bình (Mean CI)
    # Công thức: Mean ± t * (STD / sqrt(N))
    n = len(arr)
    t_value = 2.776 if n == 5 else 1.96 # t-value = 2.776 cho bậc tự do = 4 (5 folds)
    margin = t_value * (std_val / np.sqrt(n)) if n > 0 else 0.0
    
    return mean_val, std_val, mean_val - margin, mean_val + margin, max_val, max_idx
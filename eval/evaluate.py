from __future__ import annotations

import argparse
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch



PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from survival_prediction.dataset import SurvivalDataset
from survival_prediction.model import SurvivalRiskPredictor, build_feature_vector
from eval.metrics import calculate_iauc, harrell_c_index, bootstrap_c_index, calculate_cv_stats

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate survival predictions with ground truth CSV.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest file containing submitter_id values (one per line).",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("data/outputs/predictions"),
        help="Directory containing prediction outputs.",
    )
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        default=Path("data/reference/Final_Matched_Clinical.csv"),
        help="Path to the clinical ground truth CSV file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/results/cindex.csv"),
        help="Path to save the evaluation results in CSV format.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="unknown",
        help="Tên của project/cohort (VD: blca, brca) để điền vào file CSV.",
    )
    
    # Các tham số mới để chạy đánh giá 5-Fold Validation
    parser.add_argument("--five-fold", action="store_true", help="Bật chế độ đánh giá 5-Fold Validation.")
    parser.add_argument("--model-path", type=Path, default=Path("data/outputs/survival_model.pth"))
    parser.add_argument("--h5-dir", type=Path, default=Path("data/h5_feature"))
    parser.add_argument("--co-attention-dir", type=Path, default=Path("data/outputs/co_attention"))
    parser.add_argument("--fusion-dir", type=Path, default=Path("data/outputs/fusion"))
    parser.add_argument("--num-bins", type=int, default=4)
    return parser.parse_args()


def load_manifest(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            candidate = line.strip()
            if not candidate:
                continue
            if "\t" in candidate:
                parts = candidate.split("\t")
                if len(parts) >= 2:
                    filename = parts[1]
                    if len(filename) >= 12:
                        ids.append(filename[:12])
                        continue
            ids.append(candidate)
    return ids

def load_clinical_data(csv_path: Path) -> Dict[str, Tuple[float, int]]:
    """Loads ground truth from the clinical CSV file.
    
    Returns:
        A dictionary mapping submitter_id to a tuple of (OS_time_days, vital_status_binary)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {csv_path}")
        
    clinical_data = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            submitter_id = row.get("submitter_id")
            if not submitter_id:
                continue
                
            time_str = row.get("OS_time_days", "NA")
            event_str = row.get("vital_status_binary", "NA")
            
            # Chỉ lấy các bệnh nhân có đủ cả time và event (bỏ qua 'NA')
            try:
                time = float(time_str)
                event = int(event_str)
                clinical_data[submitter_id] = (time, event)
            except ValueError:
                continue  # Bỏ qua dòng này nếu dữ liệu không phải là số (VD: 'NA')
                
    return clinical_data

def format_float(val: float, precision: int = 9) -> str:
    """Helper định dạng số, trả về 'NA' nếu là NaN."""
    if np.isnan(val): return "NA"
    return f"{val:.{precision}f}"

def save_to_csv(csv_path: Path, project: str, stats: dict) -> None:
    """Ghi output vào CSV theo đúng định dạng được yêu cầu (đầy đủ các cột)."""
    headers = [
        "Project", "Project", "Val C-Index (Mean)", "Val C-Index (STD)", 
        "Val C-Index (CI)", "Val C-Index (Max)", "val_idx", "C-Index (All)", 
        "P-Value", "95% CI", "Val I-AUC (Mean)", "Val I-AUC (STD)"
    ]
    
    # Format chuỗi CI
    val_ci_str = f"({format_float(stats['val_c_ci_low'], 3)}-{format_float(stats['val_c_ci_up'], 3)})" if not np.isnan(stats['val_c_ci_low']) else "NA"
    all_ci_str = f"({format_float(stats['all_ci_low'], 3)}-{format_float(stats['all_ci_up'], 3)})" if not np.isnan(stats['all_ci_low']) else "NA"
    val_idx_str = str(stats['val_idx']) if stats['val_idx'] >= 0 else "NA"

    data_row = [
        project.upper(), project.upper(), 
        format_float(stats['val_c_mean']), 
        format_float(stats['val_c_std']), 
        val_ci_str, 
        format_float(stats['val_c_max']), 
        val_idx_str, 
        format_float(stats['c_all']), 
        format_float(stats['p_value']), 
        all_ci_str, 
        format_float(stats['val_iauc_mean']), 
        format_float(stats['val_iauc_std'])
    ]
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data_row)
        
    print(f"[SUCCESS] Đã lưu kết quả hoàn chỉnh vào: {csv_path}")
    
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clinical_data = load_clinical_data(args.clinical_csv)
    
    # Dictionary chứa kết quả thống kê
    stats = {
        'val_c_mean': np.nan, 'val_c_std': np.nan, 'val_c_ci_low': np.nan, 'val_c_ci_up': np.nan,
        'val_c_max': np.nan, 'val_idx': -1, 'c_all': np.nan, 'p_value': np.nan,
        'all_ci_low': np.nan, 'all_ci_up': np.nan, 'val_iauc_mean': np.nan, 'val_iauc_std': np.nan
    }

    if args.five_fold:
        print("=== BẮT ĐẦU ĐÁNH GIÁ 5-FOLD CROSS-VALIDATION ===")
        dataset = SurvivalDataset(
            manifest_path=args.manifest, 
            # h5_dir=args.h5_dir, 
            # co_attention_dir=args.co_attention_dir, 
            fusion_dir=args.fusion_dir
        )
        
        # Tái tạo lại phép chia fold hệt như main.py
        indices = np.arange(len(dataset))
        rng = np.random.default_rng(42)
        rng.shuffle(indices)
        folds = np.array_split(indices, 5)
        
        # Khởi tạo kích thước input = 1024 (fusion)
        first_feat = build_feature_vector(
            # dataset.samples[0].h5_features,
            # dataset.samples[0].co_attention, 
            dataset.samples[0].fusion
        )
        input_dim = first_feat.shape[0]
        
        fold_c_indices, fold_iaucs = [], []
        oof_times, oof_events, oof_scores = [], [], []
        
        for fold_idx in range(5):
            fold_path = args.model_path.parent / f"{args.model_path.stem}_fold_{fold_idx}{args.model_path.suffix}"
            if not fold_path.exists():
                raise FileNotFoundError(f"Thiếu file model Fold {fold_idx}: {fold_path}")
            
            # Load mô hình
            model = SurvivalRiskPredictor(input_dim=input_dim, num_bins=args.num_bins).to(device)
            model.load_state_dict(torch.load(fold_path, map_location=device))
            model.eval()
            
            val_indices = folds[fold_idx]
            v_times, v_events, v_scores = [], [], []
            
            with torch.no_grad():
                for idx in val_indices:
                    sample = dataset[idx]
                    sid = sample["submitter_id"]
                    if sid not in clinical_data: continue
                    
                    feat_vec = build_feature_vector(
                        # sample["h5_features"], 
                        # sample["co_attention"], 
                        sample["fusion"]
                    ).unsqueeze(0).to(device)
                    hazards = model(feat_vec).squeeze(0).cpu()
                    risk_score = float(hazards.sum().item())
                    time, event = clinical_data[sid]
                    
                    v_times.append(time); v_events.append(event); v_scores.append(risk_score)
                    oof_times.append(time); oof_events.append(event); oof_scores.append(risk_score)
            
            # Tính C-Index và I-AUC cho Fold này
            t, e, s = np.array(v_times), np.array(v_events), np.array(v_scores)
            c_idx = harrell_c_index(t, e, s)
            iauc = calculate_iauc(t, e, s)
            
            fold_c_indices.append(c_idx)
            fold_iaucs.append(iauc)
            print(f"Fold {fold_idx} | C-Index: {c_idx:.4f} | I-AUC: {iauc:.4f}")
            
        # Tổng hợp thống kê Validation
        v_mean, v_std, v_ci_low, v_ci_up, v_max, v_idx = calculate_cv_stats(fold_c_indices)
        i_mean, i_std, _, _, _, _ = calculate_cv_stats(fold_iaucs)
        
        stats.update({
            'val_c_mean': v_mean, 'val_c_std': v_std, 'val_c_ci_low': v_ci_low, 'val_c_ci_up': v_ci_up,
            'val_c_max': v_max, 'val_idx': v_idx, 'val_iauc_mean': i_mean, 'val_iauc_std': i_std
        })
        
        # Tính Out-Of-Fold (C-Index All)
        t_all, e_all, s_all = np.array(oof_times), np.array(oof_events), np.array(oof_scores)
        stats['c_all'] = harrell_c_index(t_all, e_all, s_all)
        stats['p_value'], stats['all_ci_low'], stats['all_ci_up'] = bootstrap_c_index(t_all, e_all, s_all)
        
    else:
        # CHẾ ĐỘ ĐỌC TỪ THƯ MỤC PREDICTIONS (Single/Ensemble)
        # Sẽ bỏ trống (NA) các giá trị Val
        print("=== BẮT ĐẦU ĐÁNH GIÁ TỪ THƯ MỤC PREDICTIONS ===")
        # ... (Phần code cũ đọc file .pt) ...
        # Ở chế độ này các giá trị Val sẽ được ghi là "NA" tự động
        pass 

    print("-" * 50)
    print(f"C-Index (All): {stats['c_all']:.4f}")
    if args.five_fold:
        print(f"Val C-Index Mean: {stats['val_c_mean']:.4f} ± {stats['val_c_std']:.4f}")
    print("-" * 50)
    
    save_to_csv(args.output_csv, args.project, stats)

# def main() -> None:
#     args = parse_args()
    
#     # 1. Tải danh sách bệnh nhân từ manifest
#     submitter_ids = load_manifest(args.manifest)
#     if not submitter_ids:
#         raise ValueError("Manifest contains no submitter_id values")

#     # 2. Tải Ground Truth từ file CSV
#     clinical_data = load_clinical_data(args.clinical_csv)

#     times_list: List[float] = []
#     events_list: List[int] = []
#     scores_list: List[float] = []
#     evaluated_ids: List[str] = []

#     # 3. Lặp qua danh sách và ghép (match) Prediction với Ground Truth
#     for submitter_id in submitter_ids:
#         # Kiểm tra dữ liệu clinical có tồn tại bệnh nhân này không
#         if submitter_id not in clinical_data:
#             print(f"[WARN] Missing ground truth in CSV for {submitter_id}. Skipping...")
#             continue
            
#         pred_path = args.predictions_dir / f"{submitter_id}_hazards.pt"
#         if not pred_path.exists():
#             print(f"[WARN] Missing prediction file for {submitter_id}. Skipping...")
#             continue
            
#         # Lấy prediction (Risk Score)
#         payload = torch.load(pred_path, map_location="cpu")
#         hazards = payload["hazards"]
#         risk_score = float(hazards.sum().item())
        
#         # Lấy Ground truth
#         time, event = clinical_data[submitter_id]
        
#         # Lưu vào danh sách để đánh giá
#         evaluated_ids.append(submitter_id)
#         times_list.append(time)
#         events_list.append(event)
#         scores_list.append(risk_score)

#     if not evaluated_ids:
#         raise RuntimeError("No matched predictions and ground truth found to evaluate.")

#     # 4. Chuyển sang Numpy array
#     times = np.array(times_list)
#     events = np.array(events_list)
#     scores = np.array(scores_list)

#     # 5. Tính toán chỉ số C-Index
#     c_index = harrell_c_index(times, events, scores)
    
#     # 6. Tính Time-dependent I-AUC
#     i_auc = calculate_iauc(times, events, scores)
    
#     # 7. Chạy Bootstrapping để lấy 95% CI và P-Value
#     p_value, ci_lower, ci_upper = bootstrap_c_index(times, events, scores)
    
#     print("-" * 50)
#     print(f"Total manifest requested: {len(submitter_ids)} patients")
#     print(f"Total matched & evaluated: {len(evaluated_ids)} patients")
#     print(f"C-Index (All) : {c_index:.4f}")
#     print(f"95% CI        : ({ci_lower:.4f} - {ci_upper:.4f})")
#     print(f"P-Value       : {p_value:.6f}")
#     print(f"I-AUC         : {i_auc:.4f}")
#     print("-" * 50)
    
#     # 8. Đóng gói dữ liệu vào Dictionary để tương thích với hàm save_to_csv mới
#     stats = {
#         'val_c_mean': np.nan, 
#         'val_c_std': np.nan, 
#         'val_c_ci_low': np.nan, 
#         'val_c_ci_up': np.nan,
#         'val_c_max': np.nan, 
#         'val_idx': -1, 
        
#         'c_all': c_index, 
#         'p_value': p_value,
#         'all_ci_low': ci_lower, 
#         'all_ci_up': ci_upper, 
        
#         'val_iauc_mean': i_auc, 
#         'val_iauc_std': np.nan
#     }

#     # Xuất ra file CSV
#     save_to_csv(args.output_csv, args.project, stats)
#     # # Xuất ra file CSV
#     # save_to_csv(args.output_csv, args.project, stats)
    


if __name__ == "__main__":
    main()

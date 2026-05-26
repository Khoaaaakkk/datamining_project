from __future__ import annotations

import argparse
import sys
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from survival_prediction.dataset import SurvivalDataset
from survival_prediction.model import SurvivalRiskPredictor, build_feature_vector


class NLLSurvivalLoss(nn.Module):
    """Negative Log-Likelihood Loss cho bài toán Discrete Survival Prediction."""
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, hazards: torch.Tensor, Y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        batch_size = hazards.shape[0]
        log_h = torch.log(hazards + self.eps)
        log_1_minus_h = torch.log(1.0 - hazards + self.eps)
        
        log_S = torch.cumsum(log_1_minus_h, dim=1)
        log_S_padded = torch.cat([torch.zeros(batch_size, 1, device=hazards.device), log_S], dim=1)
        
        idx = Y.view(batch_size, 1)
        uncensored_log_prob = torch.gather(log_h, 1, idx) + torch.gather(log_S_padded, 1, idx)
        censored_log_prob = torch.gather(log_S_padded, 1, idx + 1)
        
        loss = (1.0 - c) * (-uncensored_log_prob.squeeze(1)) + c * (-censored_log_prob.squeeze(1))
        return loss.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run discrete survival prediction pipeline with 5-Fold CV.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        required=True,
        help="Chế độ chạy hệ thống: 'train' để huấn luyện hoặc 'predict' để dự đoán.",
    )
    parser.add_argument(
        "--five-fold",
        action="store_true",
        help="Kích hoạt chế độ Kiểm tra chéo 5 lần (5-Fold Cross-Validation) hoặc Dự đoán Ensemble 5 mô hình.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest file containing submitter_id values (one per line).",
    )
    parser.add_argument(
        "--h5-dir",
        type=Path,
        default=Path("data/h5_feature"),
        help="Directory containing {submitter_id}.h5 files.",
    )
    parser.add_argument(
        "--co-attention-dir",
        type=Path,
        default=Path("data/outputs/co_attention"),
        help="Directory containing {submitter_id}.pt co-attention outputs.",
    )
    parser.add_argument(
        "--fusion-dir",
        type=Path,
        default=Path("data/outputs/fusion"),
        help="Directory containing {submitter_id}.pt multimodal fusion outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/predictions"),
        help="Directory to save predicted hazard tensors (Chỉ dùng cho mode predict).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/outputs/survival_model.pth"),
        help="Đường dẫn gốc lưu hoặc tải trọng số mô hình (.pth).",
    )
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        default=Path("data/reference/Final_Matched_Clinical.csv"),
        help="Đường dẫn tới file CSV chứa thông tin sống còn ground truth.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=4,
        help="Number of discrete time bins for survival prediction.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Số lượng epoch khi chạy chế độ train.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate cho optimizer.",
    )
    return parser.parse_args()


def load_clinical_data(csv_path: Path) -> Dict[str, Tuple[float, int]]:
    """Tải dữ liệu ground truth sinh tồn từ file CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file lâm sàng CSV: {csv_path}")
    
    clinical_data = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("submitter_id")
            if not sid:
                continue
            time_str = row.get("OS_time_days", "NA")
            event_str = row.get("vital_status_binary", "NA")
            try:
                time = float(time_str)
                event = int(event_str)
                clinical_data[sid] = (time, event)
            except ValueError:
                continue
    return clinical_data


def get_time_bins(clinical_data: Dict[str, Tuple[float, int]], manifest_ids: List[str], num_bins: int) -> np.ndarray:
    """Tính các mốc thời gian (bin edges) dựa trên phân vị dữ liệu của tập tương ứng."""
    times = []
    for sid in manifest_ids:
        if sid in clinical_data:
            times.append(clinical_data[sid][0])
    
    if not times:
        raise ValueError("Không tìm thấy dữ liệu thời gian hợp lệ để chia bin.")
        
    quantiles = np.linspace(0, 100, num_bins + 1)[1:-1]
    return np.percentile(times, quantiles)


def assign_bin(time: float, bin_edges: np.ndarray) -> int:
    """Xác định index của bin (từ 0 đến num_bins-1) mà thời gian rơi vào."""
    return int(np.digitize(time, bin_edges))


def collate_fn(batch: list[Dict[str, torch.Tensor]]) -> list[Dict[str, torch.Tensor]]:
    return batch


def run_single_train_loop(
    model: nn.Module, 
    train_indices: List[int], 
    dataset: SurvivalDataset, 
    clinical_data: Dict[str, Tuple[float, int]], 
    bin_edges: np.ndarray, 
    epochs: int, 
    lr: float, 
    device: torch.device,
    val_indices: List[int] | None = None
) -> None:
    """Hàm phụ trách chạy vòng lặp tối ưu hóa cho một mô hình cụ thể."""
    criterion = NLLSurvivalLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(42)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, total_samples = 0.0, 0
        
        # Xáo trộn thứ tự index ngẫu nhiên trong mỗi epoch
        shuffled_train = list(train_indices)
        rng.shuffle(shuffled_train)
        
        for idx in shuffled_train:
            sample = dataset[idx]
            sid = sample["submitter_id"]
            if sid not in clinical_data:
                continue
                
            raw_time, vital_status = clinical_data[sid]
            Y_idx = assign_bin(raw_time, bin_edges)
            c_status = 1.0 - float(vital_status)
            
            Y_tensor = torch.tensor([Y_idx], dtype=torch.long, device=device)
            c_tensor = torch.tensor([c_status], dtype=torch.float, device=device)
            
            feature_vec = build_feature_vector(
                # sample["h5_features"], 
                # sample["co_attention"], 
                sample["fusion"]
            ).unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            hazards = model(feature_vec)
            loss = criterion(hazards, Y_tensor, c_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total_samples += 1
            
        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0
        
        # Nếu có tập kiểm định song song, tính Loss Validation cuối mỗi epoch
        val_str = ""
        if val_indices is not None and epoch == epochs - 1:
            model.eval()
            val_loss, val_samples = 0.0, 0
            with torch.no_grad():
                for idx in val_indices:
                    sample = dataset[idx]
                    sid = sample["submitter_id"]
                    if sid not in clinical_data:
                        continue
                    raw_time, vital_status = clinical_data[sid]
                    Y_idx = assign_bin(raw_time, bin_edges)
                    c_status = 1.0 - float(vital_status)
                    
                    Y_tensor = torch.tensor([Y_idx], dtype=torch.long, device=device)
                    c_tensor = torch.tensor([c_status], dtype=torch.float, device=device)
                    feature_vec = build_feature_vector(
                        # sample["h5_features"], 
                        # sample["co_attention"], 
                        sample["fusion"]
                    ).unsqueeze(0).to(device)
                    
                    hazards = model(feature_vec)
                    val_loss += criterion(hazards, Y_tensor, c_tensor).item()
                    val_samples += 1
            avg_val = val_loss / val_samples if val_samples > 0 else 0
            val_str = f" | Val Loss: {avg_val:.4f}"
            
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f}{val_str}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = SurvivalDataset(
        manifest_path=args.manifest,
        # h5_dir=args.h5_dir,
        # co_attention_dir=args.co_attention_dir,
        fusion_dir=args.fusion_dir,
    )
    if len(dataset) == 0:
        raise RuntimeError("Không tìm thấy mẫu dữ liệu hợp lệ nào.")

    clinical_data = load_clinical_data(args.clinical_csv)
    manifest_ids = [s.submitter_id for s in dataset.samples]
    
    # Xác định chiều ẩn đầu vào dựa theo bài báo (luôn là 1024 từ fusion vector)
    first_feature = build_feature_vector(
        # dataset.samples[0].h5_features,
        # dataset.samples[0].co_attention,
        dataset.samples[0].fusion
    )
    input_dim = first_feature.shape[0]

    # -------------------------------------------------------------------------
    # CHẾ ĐỘ 1: HUẤN LUYỆN MÔ HÌNH (TRAIN MODE)
    # -------------------------------------------------------------------------
    if args.mode == "train":
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        num_samples = len(dataset)
        
        if args.five_fold:
            print(f"=== BẮT ĐẦU HUẤN LUYỆN CHẾ ĐỘ: 5-FOLD CROSS-VALIDATION ===")
            indices = np.arange(num_samples)
            rng = np.random.default_rng(42)  # Seed cố định để phân rã fold đồng nhất
            rng.shuffle(indices)
            folds = np.array_split(indices, 5)
            
            for fold_idx in range(5):
                print(f"\n>>> Chạy vòng huấn luyện cho Fold [{fold_idx + 1}/5]")
                val_set_indices = list(folds[fold_idx])
                val_set_hash = set(val_set_indices)
                train_set_indices = [i for i in indices if i not in val_set_hash]
                
                # Tránh data leakage: tính toán mốc thời gian rời rạc riêng trên tập huấn luyện của fold này
                train_ids = [dataset.samples[i].submitter_id for i in train_set_indices]
                fold_bin_edges = get_time_bins(clinical_data, train_ids, args.num_bins)
                
                fold_model = SurvivalRiskPredictor(input_dim=input_dim, num_bins=args.num_bins).to(device)
                
                run_single_train_loop(
                    model=fold_model,
                    train_indices=train_set_indices,
                    val_indices=val_set_indices,
                    dataset=dataset,
                    clinical_data=clinical_data,
                    bin_edges=fold_bin_edges,
                    epochs=args.epochs,
                    lr=args.lr,
                    device=device
                )
                
                # Thiết lập đường dẫn lưu file riêng cho từng fold
                fold_save_path = args.model_path.parent / f"{args.model_path.stem}_fold_{fold_idx}{args.model_path.suffix}"
                torch.save(fold_model.state_dict(), fold_save_path)
                print(f"-> Đã lưu trọng số mô hình Fold {fold_idx + 1} tại: {fold_save_path}")
        else:
            print(f"=== BẮT ĐẦU HUẤN LUYỆN CHẾ ĐỘ: SINGLE MODEL (TOÀN TẬP DATA) ===")
            global_bin_edges = get_time_bins(clinical_data, manifest_ids, args.num_bins)
            model = SurvivalRiskPredictor(input_dim=input_dim, num_bins=args.num_bins).to(device)
            
            run_single_train_loop(
                model=model,
                train_indices=list(range(num_samples)),
                val_indices=None,
                dataset=dataset,
                clinical_data=clinical_data,
                bin_edges=global_bin_edges,
                epochs=args.epochs,
                lr=args.lr,
                device=device
            )
            torch.save(model.state_dict(), args.model_path)
            print(f"-> Đã lưu mô hình đơn thành công tại: {args.model_path}")

    # -------------------------------------------------------------------------
    # CHẾ ĐỘ 2: DỰ ĐOÁN NGUY CƠ (PREDICT MODE)
    # -------------------------------------------------------------------------
    elif args.mode == "predict":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # Nạp các kiến trúc mạng nơ-ron tương ứng với tham số cấu hình
        models_pool = []
        if args.five_fold:
            print(f"=== BẮT ĐẦU CHẾ ĐỘ DỰ ĐOÁN: ENSEMBLE (GỘP KẾT QUẢ 5 FOLDS) ===")
            for fold_idx in range(5):
                fold_path = args.model_path.parent / f"{args.model_path.stem}_fold_{fold_idx}{args.model_path.suffix}"
                if not fold_path.exists():
                    raise FileNotFoundError(f"Thiếu file checkpoint {fold_path}. Vui lòng chạy train --five-fold trước.")
                m = SurvivalRiskPredictor(input_dim=input_dim, num_bins=args.num_bins).to(device)
                m.load_state_dict(torch.load(fold_path, map_location=device))
                m.eval()
                models_pool.append(m)
        else:
            print(f"=== BẮT ĐẦU CHẾ ĐỘ DỰ ĐOÁN: SINGLE MODEL ===")
            if not args.model_path.exists():
                raise FileNotFoundError(f"Thiếu file checkpoint gốc tại {args.model_path}.")
            m = SurvivalRiskPredictor(input_dim=input_dim, num_bins=args.num_bins).to(device)
            m.load_state_dict(torch.load(args.model_path, map_location=device))
            m.eval()
            models_pool.append(m)
            
        # Vòng lặp dự đoán sinh tồn trên từng mẫu dữ liệu
        with torch.no_grad():
            for batch in loader:
                sample = batch[0]
                submitter_id = sample["submitter_id"]
                
                feature_vec = build_feature_vector(
                    # sample["h5_features"], 
                    # sample["co_attention"], 
                    sample["fusion"]
                ).unsqueeze(0).to(device)
                
                # Tính giá trị trung bình từ tập hợp các mô hình được nạp
                hazards_outputs = [m(feature_vec) for m in models_pool]
                hazards = torch.stack(hazards_outputs).mean(dim=0).squeeze(0).cpu()
                
                output_path = args.output_dir / f"{submitter_id}_hazards.pt"
                torch.save({"hazards": hazards}, output_path)
                print(f"Saved {'Ensemble ' if args.five_fold else ''}hazards for {submitter_id} -> {output_path}")


if __name__ == "__main__":
    main()
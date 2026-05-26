import torch
import torch.nn as nn
import pandas as pd
import pickle
from pathlib import Path

# --- LỚP MÔ HÌNH GENENCODER ---
class GenEncoder(nn.Module):
    def __init__(self, bag_info):
        super(GenEncoder, self).__init__()
        self.encoders = nn.ModuleDict()
        for name, num_genes in bag_info.items():
            self.encoders[name] = nn.Sequential(
                nn.BatchNorm1d(num_genes), 
                nn.Linear(num_genes, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512)
            )

    def forward(self, x_dict):
        output_vectors = {}
        for name, data in x_dict.items():
            output_vectors[name] = self.encoders[name](data)
        return output_vectors

# --- CẤU HÌNH ---
data_dir = Path("../data/csv")
bag_files = {
    'CellCycle': data_dir / 'bag_CellCycle.csv',
    'DNADamage': data_dir / 'bag_DNADamage.csv',
    'EMT': data_dir / 'bag_EMT.csv',
    'Hormone': data_dir / 'bag_Hormone.csv',
    'Immune': data_dir / 'bag_Immune.csv',
    'Other': data_dir / 'bag_Other.csv'
}

bag_info = {}
data_dict = {}

# --- BƯỚC 1: ĐỌC DỮ LIỆU ---
submitter_ids = None
for name, file in bag_files.items():
    df = pd.read_csv(file)
    if submitter_ids is None:
        submitter_ids = df['submitter_id']
    
    num_genes = df.shape[1] - 1
    bag_info[name] = num_genes
    data_dict[name] = torch.tensor(df.drop(columns=['submitter_id']).values, dtype=torch.float32)

# --- BƯỚC 2: KHỞI TẠO VÀ CHẠY DỮ LIỆU QUA MODEL ---
model = GenEncoder(bag_info)
model.eval() # Chế độ trích xuất đặc trưng

with torch.no_grad():
    # ĐÂY LÀ BƯỚC BẠN THIẾU: Phải chạy dữ liệu qua model
    encoded_outputs = model(data_dict)

# --- BƯỚC 3: XUẤT 6 FILE RIÊNG BIỆT ---
print("Đang xuất 6 file riêng biệt...")
data_dir.mkdir(parents=True, exist_ok=True)
for name, vec in encoded_outputs.items():
    encoded_df = pd.DataFrame(vec.numpy(), columns=[f'feat_{i}' for i in range(512)])
    encoded_df.insert(0, 'submitter_id', submitter_ids)
    output_path = data_dir / f"encoded_{name}.csv"
    encoded_df.to_csv(output_path, index=False)
    print(f"Đã lưu: {output_path}")

# --- BƯỚC 4: STACK THÀNH [N or 6, 512] ---
# Dùng thứ tự cố định để đảm bảo 6 túi luôn ở vị trí giống nhau
ordered_keys = ['CellCycle', 'DNADamage', 'EMT', 'Hormone', 'Immune', 'Other']
tensors_list = [encoded_outputs[k] for k in ordered_keys]

combined_matrix = torch.stack(tensors_list, dim=1)

print(f"\nShape cuối cùng của ma trận: {combined_matrix.shape}")

# --- BƯỚC 5: LƯU MA TRẬN 3D ---
combined_path = data_dir / "combined_genomic_features.pkl"
with open(combined_path, "wb") as f:
    pickle.dump(combined_matrix, f)

print(f"Đã lưu ma trận [N, 6, 512] vào '{combined_path}'")

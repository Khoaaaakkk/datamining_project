import pandas as pd
import mygene
import requests
import gseapy as gp
from collections import defaultdict

# Cấu hình
TCGA_FILE = "../data/csv/Python_Input_Genomic_LogCPM.csv" 
 
# 1. Đọc dữ liệu 
import time # Nhớ import thư viện time ở đầu file nhé

# 1. Đọc dữ liệu
print("\n" + "="*50)
print("Đang nạp ma trận khổng lồ (Fat Matrix) vào RAM...")  

start_time = time.time()

# Dùng bộ máy mặc định, tắt low_memory để nhồi thẳng toàn bộ vào RAM
df = pd.read_csv(TCGA_FILE, index_col=0, low_memory=False)

end_time = time.time()
print(f"\n-> THÀNH CÔNG! Đã nạp xong trong {round(end_time - start_time, 1)} giây.")
print(f"Kích thước dữ liệu: {df.shape[0]} bệnh nhân, {df.shape[1]} gene.")

# 2. Xử lý Ensembl ID version
ens_to_full = {}
for full_id in df.columns:
    base = full_id.split('.')[0]
    ens_to_full[full_id] = base
base_ids = list(set(ens_to_full.values()))
print(f"Số Ensembl ID duy nhất (bỏ version): {len(base_ids)}")

# 3. Chuyển đổi sang Symbol
mg = mygene.MyGeneInfo()
print("Đang query mygene (có thể mất vài phút)...")
results = mg.querymany(base_ids, scopes='ensembl.gene', fields='symbol', species='human', returnall=True)
base_to_sym = {}
for item in results['out']:
    if 'symbol' in item and item['symbol']:
        base_to_sym[item['query']] = item['symbol']
print(f"Đã ánh xạ {len(base_to_sym)}/{len(base_ids)} base Ensembl ID thành công.")

# Gán symbol cho các cột có version
full_to_sym = {}
for full, base in ens_to_full.items():
    if base in base_to_sym:
        full_to_sym[full] = base_to_sym[base]

mapped_columns = list(full_to_sym.keys())
df_mapped = df[mapped_columns].copy()
df_mapped.columns = [full_to_sym[col] for col in mapped_columns]

# Gộp các cột cùng symbol
print("Gộp các Ensembl ID cùng symbol (tính trung bình)...")
df_symbol = df_mapped.T.groupby(level=0).mean().T
print(f"Ma trận cuối cùng với Symbol: {df_symbol.shape[1]} gene.")

# 4. Lấy Hallmark gene sets từ MSigDB bằng gseapy
print("\nĐang kết nối với MSigDB qua gseapy...")
msig = gp.Msigdb()  # tạo instance Msigdb

# Lấy các bộ Hallmark (thử dbver="2026.1.Hs", nếu lỗi sẽ tự động chọn bản mới nhất)
try:
    hallmark_sets = msig.get_gmt(category='h.all', dbver="2026.1.Hs")
    print("Dùng dbver=2026.1.Hs")
except Exception:
    # fallback về phiên bản mới nhất có sẵn
    hallmark_sets = msig.get_gmt(category='h.all')
    print("Dùng bản mới nhất có sẵn")

# hallmark_sets là dict dạng {pathway_name: [gene_symbol1, gene_symbol2, ...]}
print(f"Đã tải {len(hallmark_sets)} bộ Hallmark.")

# 5. Phân loại thành 6 nhóm theo từ khóa
def classify_pathway(name):
    n = name.upper()
    if any(k in n for k in ['E2F', 'G2M', 'MITOTIC']):
        return 'CellCycle'
    if any(k in n for k in ['DNA_REPAIR', 'P53', 'UV_RESPONSE']):
        return 'DNADamage'
    if any(k in n for k in ['EPITHELIAL_MESENCHYMAL', 'EMT', 'ANGIOGENESIS', 'APICAL', 'HEDGEHOG']):
        return 'EMT'
    if any(k in n for k in ['ESTROGEN', 'ANDROGEN']):
        return 'Hormone'
    if any(k in n for k in ['INFLAMMATORY', 'INTERFERON', 'TNFA', 'IL6', 'JAK',
                             'ALLOGRAFT', 'COMPLEMENT', 'COAGULATION']):
        return 'Immune'
    return 'Other'

group_genes = defaultdict(set)
for pw, genes in hallmark_sets.items():
    group_genes[classify_pathway(pw)].update(genes)

print("Các nhóm và số gene (trước lọc):")
for grp, genes in group_genes.items():
    print(f"  {grp}: {len(genes)} gene")

# 6. Tạo bag cho từng nhóm
bag_data = {}
for grp, symbols in group_genes.items():
    available = [s for s in symbols if s in df_symbol.columns]
    if not available:
        print(f"Bag '{grp}': không có gene nào trong dữ liệu bệnh nhân.")
        continue
    bag_df = df_symbol[available].reset_index()
    bag_df.rename(columns={bag_df.columns[0]: 'submitter_id'}, inplace=True)
    bag_data[grp] = bag_df
    print(f"Bag '{grp}': {bag_df.shape[1]-1} gene, {bag_df.shape[0]} bệnh nhân")

# 7. Lưu file CSV
for grp, bag in bag_data.items():
    fname = f"../data/csv/bag_{grp}.csv"
    bag.to_csv(fname, index=False)
    print(f"Đã lưu {fname}")

print("Hoàn tất!")




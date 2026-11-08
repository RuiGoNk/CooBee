import csv
import os
import random
import statistics

BASE_DIR = "C:/Users/79688/PycharmProjects/CooBee_Research/"

def read_ndcg(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath): return []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        return [float(r["ndcg"]) for r in csv.DictReader(f)]

def bootstrap_ci(data, num_samples=10000, confidence=0.95):
    if not data: return 0, 0
    n = len(data)
    means = []
    random.seed(42)
    for _ in range(num_samples):
        sample = [random.choice(data) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_idx = int((1 - confidence) / 2 * num_samples)
    upper_idx = int((1 + confidence) / 2 * num_samples)
    return means[lower_idx], means[upper_idx]

configs = {
    "B0 (Rules)": read_ndcg("results_b0.csv"),
    "B1 (Single-shot)": read_ndcg("results_b1.csv"),
    "H1 (Constraint)": read_ndcg("results_h1_constraint.csv"),
    "H2 (Few-Shot)": read_ndcg("results_h2_fewshot.csv"),
    "H3 (Pro)": read_ndcg("results_h3_pro.csv"),
    "H4 (Math Rerank)": read_ndcg("results_h4_mathrerank.csv"),
    "H5 (Weighted Rerank)": read_ndcg("results_h5.csv")
}

print(f"{'Конфигурация':<25} | {'NDCG@5':<8} | {'95% CI':<20}")
print("-" * 60)
for name, data in configs.items():
    if data:
        mean = statistics.mean(data)
        low, high = bootstrap_ci(data)
        print(f"{name:<25} | {mean:.4f}   | [{low:.4f}, {high:.4f}]")

# Проверка значимости H5 vs B0
h5 = configs["H5 (Weighted Rerank)"]
b0 = configs["B0 (Rules)"]
if h5 and b0:
    h5_low, h5_high = bootstrap_ci(h5)
    b0_low, b0_high = bootstrap_ci(b0)
    print("\n--- Сравнение H5 vs B0 ---")
    if h5_low > b0_high:
        print("Разница значима (CI не пересекаются).")
    else:
        print("Разница НЕ значима на n=25 (CI пересекаются). Требуется расширение выборки.")

# Аномалии H5
print("\n--- Аномалии H5 ---")
h5_non_zero = [x for x in h5 if x > 0]
print(f"Всего запросов: {len(h5)}, NDCG = {statistics.mean(h5):.4f}")
print(f"Без аномалий (NDCG>0): {len(h5_non_zero)}, NDCG = {statistics.mean(h5_non_zero):.4f}")
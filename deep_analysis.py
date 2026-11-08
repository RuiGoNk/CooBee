import csv
import os
import math
import random
import statistics

BASE_DIR = "C:/Users/79688/PycharmProjects/CooBee_Research/"
CATALOG_FILE = os.path.join(BASE_DIR, "catalog_research_v4.csv")
GOLDEN_FILE = os.path.join(BASE_DIR, "golden_set.csv")


def load_catalog():
    with open(CATALOG_FILE, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_golden():
    with open(GOLDEN_FILE, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        for r in rows:
            r["relevant_dict"] = {}
            for item in r["relevant_ids_with_grades"].split("|"):
                item_id, grade = item.split(":")
                r["relevant_dict"][item_id] = int(grade)
        return rows


def calculate_ndcg(recommended, relevant_dict, k=5):
    dcg = 0.0
    for i, item_id in enumerate(recommended[:k]):
        grade = relevant_dict.get(item_id, 0)
        dcg += (2 ** grade - 1) / math.log2(i + 2)
    ideal_grades = sorted(relevant_dict.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    return dcg / idcg if idcg > 0 else 0.0


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


def read_results(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath): return {}
    with open(filepath, "r", encoding="utf-8-sig") as f:
        return {r["query_id"]: r for r in csv.DictReader(f)}


def main():
    catalog = load_catalog()
    golden = load_golden()

    # 1. Пересчет B0 на 32 запросах
    b0_results = {}
    for q in golden:
        q_id = q["query_id"]
        b_min, b_max = int(q["budget_min"]), int(q["budget_max"])
        interests = set(q["interests"].split("|"))

        candidates = []
        for item in catalog:
            p_min, p_max = int(item["price_min"]), int(item["price_max"])
            if p_max < b_min or p_min > b_max: continue
            if q["occasion"] not in item["occasions"].split("|"): continue
            tag_overlap = len(interests.intersection(set(item["tags"].split("|"))))
            if "any" in interests: tag_overlap = 1
            if tag_overlap > 0:
                candidates.append({"id": item["id"], "score": tag_overlap, "rating": float(item["rating"])})

        candidates.sort(key=lambda x: (x["score"], x["rating"]), reverse=True)
        final_recs = [c["id"] for c in candidates[:5]]
        if len(final_recs) < 5:
            for item in catalog:
                if item["id"] not in final_recs: final_recs.append(item["id"])
                if len(final_recs) == 5: break

        b0_results[q_id] = calculate_ndcg(final_recs, q["relevant_dict"])

    # 2. Загрузка B1, H5, H6
    b1_data = read_results("results_b1.csv")
    h5_data = read_results("results_h5.csv")
    h6_data = read_results("results_h6.csv")

    b1_ndcg = {q: float(r["ndcg"]) for q, r in b1_data.items()}
    h5_ndcg = {q: float(r["ndcg"]) for q, r in h5_data.items()}
    h6_ndcg = {q: float(r["ndcg"]) for q, r in h6_data.items()}

    # 3. Страты
    obvious = [f"q_{i:03d}" for i in range(1, 26)]
    non_obvious = [f"q_{i:03d}" for i in range(26, 33)]

    print("\n--- 1. ПЕРЕСЧЕТ СРЕДНИХ НА 32 ЗАПРОСАХ ---")
    all_b0 = list(b0_results.values())
    all_b1 = list(b1_ndcg.values())
    all_h5 = list(h5_ndcg.values())
    all_h6 = list(h6_ndcg.values())

    print(f"B0 (32 запроса): {statistics.mean(all_b0):.4f}")
    print(f"B1 (32 запроса): {statistics.mean(all_b1):.4f}")
    print(f"H5 (32 запроса): {statistics.mean(all_h5):.4f}")
    print(f"H6 (32 запроса): {statistics.mean(all_h6):.4f}")

    print("\n--- 2. ДЕТАЛЬНЫЙ РАЗБОР B1 НА НЕОЧЕВИДНЫХ (n=7) ---")
    for q in non_obvious:
        if q not in b1_data: continue
        raw_recs = b1_data[q].get("recs", "").split("|")
        # Проверяем, сколько из них валидны (есть в каталоге)
        valid_count = sum(1 for r in raw_recs if r in [c["id"] for c in catalog])
        print(f"{q}: NDCG={b1_ndcg[q]:.4f} | Валидных ID: {valid_count}/5 | Recs: {raw_recs}")

    print("\n--- 3. ПОЧЕМУ H6 > H5 (СРАВНЕНИЕ TOP-5) ---")
    for q in non_obvious:
        if q not in h5_data or q not in h6_data: continue
        h5_recs = h5_data[q].get("recs", "").split("|")
        h6_recs = h6_data[q].get("recs", "").split("|")
        print(f"{q}: H5({h5_ndcg[q]:.2f}) {h5_recs} | H6({h6_ndcg[q]:.2f}) {h6_recs}")

    print("\n--- 4. BOOTSTRAP CI ДЛЯ ВСЕХ КОНФИГУРАЦИЙ ---")
    for name, data in [("B0", all_b0), ("B1", all_b1), ("H5", all_h5), ("H6", all_h6)]:
        low, high = bootstrap_ci(data)
        print(f"{name}: {statistics.mean(data):.4f} [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    main()
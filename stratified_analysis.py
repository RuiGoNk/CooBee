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


def generate_b0(catalog, golden_set):
    results = []
    for q in golden_set:
        q_id = q["query_id"]
        b_min, b_max = int(q["budget_min"]), int(q["budget_max"])
        interests = set(q["interests"].split("|"))

        candidates = []
        for item in catalog:
            p_min, p_max = int(item["price_min"]), int(item["price_max"])
            if p_max < b_min or p_min > b_max: continue
            if q["occasion"] not in item["occasions"].split("|"): continue

            tag_overlap = len(interests.intersection(set(item["tags"].split("|"))))
            # Если интересы "any", даем базовый скор, чтобы не отсечь все
            if "any" in interests:
                tag_overlap = 1

            if tag_overlap > 0:
                candidates.append({"id": item["id"], "score": tag_overlap, "rating": float(item["rating"])})

        candidates.sort(key=lambda x: (x["score"], x["rating"]), reverse=True)
        final_recs = [c["id"] for c in candidates[:5]]

        if len(final_recs) < 5:
            for item in catalog:
                if item["id"] not in final_recs: final_recs.append(item["id"])
                if len(final_recs) == 5: break

        ndcg = calculate_ndcg(final_recs, q["relevant_dict"])
        results.append({"query_id": q_id, "ndcg": ndcg})
    return results


def read_ndcg(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath): return []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        return [(r["query_id"], float(r["ndcg"])) for r in rows]


def main():
    catalog = load_catalog()
    golden = load_golden()

    # 1. Генерируем B0 на всех 32 запросах
    print("Генерирую B0 на 32 запросах...")
    b0_data = generate_b0(catalog, golden)
    b0_dict = {r["query_id"]: r["ndcg"] for r in b0_data}

    # 2. Загружаем результаты B1 и H5
    b1_data = dict(read_ndcg("results_b1.csv"))
    h5_data = dict(read_ndcg("results_h5.csv"))

    # 3. Разделяем на страты
    obvious = [f"q_{i:03d}" for i in range(1, 26)]
    non_obvious = [f"q_{i:03d}" for i in range(26, 33)]

    def calc_strat_ndcg(data_dict, queries):
        vals = [data_dict[q] for q in queries if q in data_dict]
        return statistics.mean(vals) if vals else 0.0

    print("\n--- РЕЗУЛЬТАТЫ ПО СТРАТАМ ---")
    print(f"{'Конфигурация':<20} | {'Очевидные (n=25)':<18} | {'Неочевидные (n=7)':<18}")
    print("-" * 65)

    for name, data in [("B0 (Правила)", b0_dict), ("B1 (Single-shot)", b1_data), ("H5 (Harness)", h5_data)]:
        ob_ndcg = calc_strat_ndcg(data, obvious)
        nob_ndcg = calc_strat_ndcg(data, non_obvious)
        print(f"{name:<20} | {ob_ndcg:.4f}             | {nob_ndcg:.4f}")


if __name__ == "__main__":
    main()
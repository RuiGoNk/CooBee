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
    ideal_grades = sorted([g for g in relevant_dict.values() if g > 0], reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    return dcg / idcg if idcg > 0 else 0.0


# --- ГЕНЕРАЦИЯ B0 (Бизнес-правила) ---
def generate_b0():
    catalog = load_catalog()
    golden_set = load_golden()
    results = []

    for q in golden_set:
        q_id = q["query_id"]
        b_min, b_max = int(q["budget_min"]), int(q["budget_max"])
        interests = set(q["interests"].split("|"))

        candidates = []
        for item in catalog:
            p_min, p_max = int(item["price_min"]), int(item["price_max"])
            if p_max >= b_min and p_min <= b_max:
                if q["occasion"] in item["occasions"].split("|"):
                    tag_overlap = len(interests.intersection(set(item["tags"].split("|"))))
                    if tag_overlap > 0:
                        candidates.append({"id": item["id"], "score": tag_overlap, "rating": float(item["rating"])})

        candidates.sort(key=lambda x: (x["score"], x["rating"]), reverse=True)
        final_recs = [c["id"] for c in candidates[:5]]

        if len(final_recs) < 5:
            for item in catalog:
                if item["id"] not in final_recs:
                    final_recs.append(item["id"])
                if len(final_recs) == 5: break

        ndcg = calculate_ndcg(final_recs, q["relevant_dict"])
        results.append({"query_id": q_id, "ndcg": ndcg})

    with open(os.path.join(BASE_DIR, "results_b0.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "ndcg"])
        writer.writeheader()
        writer.writerows(results)
    return results


# --- BOOTSTRAP CI ---
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


def read_ndcg(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath): return []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        return [float(r["ndcg"]) for r in csv.DictReader(f)]

h6 = read_ndcg("results_h6.csv")
if h6:
    print(f"H6 (Diversified): NDCG@5 = {statistics.mean(h6):.4f}")
    # Для Serendipity нужно прочитать golden_set и проверить unobvious_id

if __name__ == "__main__":
    print("1. Генерирую результаты для B0 (Бизнес-правила)...")
    b0_data = generate_b0()
    b0_ndcg = [r["ndcg"] for r in b0_data]
    print(f"   B0 NDCG@5 = {statistics.mean(b0_ndcg):.4f}")

    print("\n2. Загружаю результаты экспериментов...")
    b1 = read_ndcg("results_b1.csv")
    h5 = read_ndcg("results_h5.csv")

    if not b1 or not h5:
        print(
            "   ВНИМАНИЕ: Не найдены файлы results_b1.csv или results_h5.csv. Сначала запустите auto_save_results.py!")
    else:
        print(f"   B1 NDCG@5 = {statistics.mean(b1):.4f}")
        print(f"   H5 NDCG@5 = {statistics.mean(h5):.4f}")

        print("\n3. Считаю Bootstrap 95% CI (доверительные интервалы)...")
        b0_low, b0_high = bootstrap_ci(b0_ndcg)
        b1_low, b1_high = bootstrap_ci(b1)
        h5_low, h5_high = bootstrap_ci(h5)

        print(f"   B0: [{b0_low:.4f}, {b0_high:.4f}]")
        print(f"   B1: [{b1_low:.4f}, {b1_high:.4f}]")
        print(f"   H5: [{h5_low:.4f}, {h5_high:.4f}]")

        print("\n4. Проверка статистической значимости (H5 vs B1):")
        if h5_low > b1_high:
            print("   ✅ РАЗНИЦА СТАТИСТИЧЕСКИ ЗНАЧИМА (интервалы не пересекаются).")
        else:
            print("   ⚠️ Интервалы пересекаются. Требуется больше данных.")

        print("\n5. Анализ NDCG без аномальных запросов (NDCG=0):")
        h5_non_zero = [x for x in h5 if x > 0]
        print(f"   H5 на всех 25 запросах: {statistics.mean(h5):.4f}")
        print(f"   H5 на {len(h5_non_zero)} запросах (без нулей): {statistics.mean(h5_non_zero):.4f}")

        print("\n--- ИТОГОВАЯ ТАБЛИЦА ДЛЯ ДИПЛОМА ---")
        print(f"{'Конфигурация':<20} | {'NDCG@5':<8} | {'95% CI':<15}")
        print("-" * 50)
        print(f"{'B0 (Rules)':<20} | {statistics.mean(b0_ndcg):.4f}   | [{b0_low:.4f}, {b0_high:.4f}]")
        print(f"{'B1 (Single-shot)':<20} | {statistics.mean(b1):.4f}   | [{b1_low:.4f}, {b1_high:.4f}]")
        print(f"{'H5 (Final Harness)':<20} | {statistics.mean(h5):.4f}   | [{h5_low:.4f}, {h5_high:.4f}]")
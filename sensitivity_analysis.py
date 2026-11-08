import csv
import os
import math
import time
import urllib3
import requests
import uuid
import re
from dotenv import load_dotenv

load_dotenv()

AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")

if not AUTH_KEY:
    print("ОШИБКА: Ключ GIGACHAT_AUTH_KEY не найден. Проверьте файл .env")
    exit()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = "C:/Users/79688/PycharmProjects/CooBee_Research/"
CATALOG_FILE = os.path.join(BASE_DIR, "catalog_research_v4.csv")
GOLDEN_FILE = os.path.join(BASE_DIR, "golden_set.csv")

def get_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': f'Basic {AUTH_KEY}'
    }
    response = requests.post(url, headers=headers, data={'scope': 'GIGACHAT_API_PERS'}, verify=False)
    return response.json().get('access_token') if response.status_code == 200 else None


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


def call_llm(token, prompt, model="GigaChat-Pro"):
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 300
    }
    response = requests.post(url, headers=headers, json=payload, verify=False)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    return ""


def run_sensitivity(weights, token, catalog, golden_set):
    w_llm, w_tags, w_rating = weights
    print(f"\n--- Тест весов: LLM={w_llm}, Теги={w_tags}, Рейтинг={w_rating} ---")
    total_ndcg = 0.0

    for q in golden_set:
        b_min, b_max = int(q["budget_min"]), int(q["budget_max"])

        valid_candidates = []
        for item in catalog:
            p_min, p_max = int(item["price_min"]), int(item["price_max"])
            if p_max >= b_min and p_min <= b_max and q["occasion"] in item["occasions"].split("|"):
                valid_candidates.append(item)
        if len(valid_candidates) < 10:
            b_min_r, b_max_r = int(b_min * 0.7), int(b_max * 1.3)
            valid_candidates = [i for i in catalog if int(i["price_max"]) >= b_min_r and int(i["price_min"]) <= b_max_r]
            if not valid_candidates: valid_candidates = catalog

        cands = "\n".join([f"{c['id']} | {c['title']} | Теги: {c['tags']}" for c in valid_candidates])
        prompt = f"""Выбери 10 ID и оцени релевантность от 1 до 10.
Запрос: {q['interests']} для {q['recipient_types']}, повод: {q['occasion']}.
Список:\n{cands}\nВерни строго JSON-массив объектов: [{{"id": "item_XXX", "score": 9}}, ...]"""

        raw_text = call_llm(token, prompt)
        found_ids = re.findall(r'item_\d{3}', raw_text)

        unique_recs = []
        for r in found_ids:
            if r in [c["id"] for c in valid_candidates] and r not in unique_recs:
                unique_recs.append(r)

        interests = set(q["interests"].split("|"))
        scored_pool = []
        for cid in unique_recs:
            item = next(c for c in catalog if c["id"] == cid)
            tag_overlap = len(interests.intersection(set(item["tags"].split("|"))))
            C, m = 50, 4.2
            rating = float(item["rating"])
            reviews = int(item["reviews_count"])
            bayesian_rating = (C * m + reviews * rating) / (C + reviews)

            match = re.search(rf'{cid}[^0-9]*(\d+)', raw_text)
            llm_score = int(match.group(1)) if match else 0

            final_score = (llm_score * w_llm) + (tag_overlap * w_tags) + (bayesian_rating * w_rating)
            scored_pool.append({"id": cid, "score": final_score})

        scored_pool.sort(key=lambda x: x["score"], reverse=True)
        final_recs = [x["id"] for x in scored_pool][:5]

        if len(final_recs) < 5:
            for cand in sorted(valid_candidates, key=lambda x: float(x["rating"]), reverse=True):
                if cand["id"] not in final_recs: final_recs.append(cand["id"])
                if len(final_recs) == 5: break

        ndcg = calculate_ndcg(final_recs, q["relevant_dict"])
        total_ndcg += ndcg
        time.sleep(1)

    mean_ndcg = total_ndcg / len(golden_set)
    print(f"  => NDCG@5 = {mean_ndcg:.4f}")
    return mean_ndcg


if __name__ == "__main__":
    catalog, golden_set = load_catalog(), load_golden()
    token = get_token()

    # 3 набора весов
    results = []
    results.append(run_sensitivity((0.5, 1.5, 1.0), token, catalog, golden_set))  # Базовые
    results.append(run_sensitivity((0.3, 1.7, 1.0), token, catalog, golden_set))  # Меньше LLM
    results.append(run_sensitivity((0.7, 1.3, 1.0), token, catalog, golden_set))  # Больше LLM

    print("\n--- ИТОГ SENSITIVITY ANALYSIS ---")
    print(f"Базовые (0.5/1.5/1.0): {results[0]:.4f}")
    print(f"Сдвиг к тегам (0.3/1.7/1.0): {results[1]:.4f}")
    print(f"Сдвиг к LLM (0.7/1.3/1.0): {results[2]:.4f}")

    # Проверка стабильности
    mean_res = sum(results) / 3
    max_dev = max(abs(r - mean_res) for r in results)
    print(f"Среднее: {mean_res:.4f}")
    print(f"Максимальное отклонение: {max_dev:.4f}")
    if max_dev < 0.05:
        print("✅ Результат стабилен. Веса не подогнаны.")
    else:
        print("⚠️ Результат чувствителен к весам.")
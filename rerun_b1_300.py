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
    ideal_grades = sorted([g for g in relevant_dict.values() if g > 0], reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    return dcg / idcg if idcg > 0 else 0.0


def call_llm(token, prompt, model="GigaChat"):
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


def main():
    catalog, golden_set = load_catalog(), load_golden()
    token = get_token()
    catalog_ids = {item["id"] for item in catalog}

    catalog_text = "".join(
        [f"ID: {i['id']} | {i['title']} | Цена: {i['price_min']}-{i['price_max']} | Теги: {i['tags']}\n" for i in
         catalog])

    total_ndcg = 0.0
    results_list = []

    print(f"Перепрогон B1 с max_tokens=300 (всего {len(golden_set)} запросов)...")
    for q in golden_set:
        q_id = q["query_id"]
        prompt = f"""Ты — ассистент по подбору подарков.
Вот запрос: Повод: {q['occasion']}, Кому: {q['recipient_types']}, Бюджет: {q['budget_min']}-{q['budget_max']}, Интересы: {q['interests']}.

Каталог:
{catalog_text}

Выбери ровно 5 ID из каталога. Верни ТОЛЬКО JSON-массив: ["item_XXX", ...]"""

        raw_text = call_llm(token, prompt)
        found_ids = re.findall(r'item_\d{3}', raw_text)

        final_recs = []
        for r in found_ids:
            if r in catalog_ids and r not in final_recs:
                final_recs.append(r)

        if len(final_recs) < 5:
            for c in catalog:
                if c["id"] not in final_recs: final_recs.append(c["id"])
                if len(final_recs) == 5: break

        ndcg = calculate_ndcg(final_recs, q["relevant_dict"])
        total_ndcg += ndcg
        results_list.append({"query_id": q_id, "ndcg": ndcg, "recs": "|".join(final_recs)})
        time.sleep(1)

    mean_ndcg = total_ndcg / len(golden_set)
    print(f"=> NDCG@5 (max_tokens=300) = {mean_ndcg:.4f}")

    # СОХРАНЯЕМ В CSV
    out_file = os.path.join(BASE_DIR, "results_b1.csv")
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "ndcg", "recs"])
        writer.writeheader()
        writer.writerows(results_list)
    print(f"Результаты сохранены в {out_file}")


if __name__ == "__main__":
    main()
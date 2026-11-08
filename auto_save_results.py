import csv
import os
import math
import time
import urllib3
import requests
import uuid
import re
import json
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


def prompt_h1(q, candidates, b_min, b_max, top_k):
    cands = "\n".join([f"{c['id']} | {c['title']} | Теги: {c['tags']}" for c in candidates])
    return f"""Ты — эксперт по подбору подарков. 
Запрос: {q['interests']} для {q['recipient_types']}, повод: {q['occasion']}, бюджет: {b_min}-{b_max}.
ПРАВИЛА: 1. Товар должен строго соответствовать интересам. 2. Не выбирай нерелевантное.
Список товаров:\n{cands}\nВерни ТОЛЬКО JSON-массив из {top_k} ID."""


def prompt_h2(q, candidates, b_min, b_max, top_k):
    cands = "\n".join([f"{c['id']} | {c['title']} | Теги: {c['tags']}" for c in candidates])
    return f"""Ты — рекомендательный алгоритм. Пример: "кулинария". Список: ["item_005 | Ножи | cooking", "item_012 | Коврик | fitness"]. Ответ: ["item_005"].
Теперь твой запрос: {q['interests']} для {q['recipient_types']}, повод: {q['occasion']}.
Список:\n{cands}\nВерни ТОЛЬКО JSON-массив из {top_k} ID."""


def prompt_h4(q, candidates, b_min, b_max, top_k):
    cands = "\n".join([f"{c['id']} | {c['title']} | Теги: {c['tags']}" for c in candidates])
    return f"""Выбери {top_k} ID. Запрос: {q['interests']} для {q['recipient_types']}, повод: {q['occasion']}.
Список:\n{cands}\nВерни ТОЛЬКО JSON-массив ID."""


def prompt_h5(q, candidates, b_min, b_max, top_k):
    cands = "\n".join([f"{c['id']} | {c['title']} | Теги: {c['tags']}" for c in candidates])
    return f"""Выбери {top_k} ID и оцени релевантность от 1 до 10.
Запрос: {q['interests']} для {q['recipient_types']}, повод: {q['occasion']}.
Список:\n{cands}\nВерни строго JSON-массив объектов: [{{"id": "item_XXX", "score": 9}}, ...]"""


def run_and_save(config_name, golden_set, catalog, token, prompt_func, model="GigaChat", top_k_llm=5):
    print(f"\n--- Запуск {config_name} ---")
    total_ndcg = 0.0
    total_serendipity = 0
    results_list = []

    for q in golden_set:
        q_id = q["query_id"]
        b_min, b_max = int(q["budget_min"]), int(q["budget_max"])
        interests = set(q["interests"].split("|"))

        valid_candidates = []
        for item in catalog:
            p_min, p_max = int(item["price_min"]), int(item["price_max"])
            if p_max >= b_min and p_min <= b_max and q["occasion"] in item["occasions"].split("|"):
                valid_candidates.append(item)
        if len(valid_candidates) < 10:
            b_min_r, b_max_r = int(b_min * 0.7), int(b_max * 1.3)
            valid_candidates = [i for i in catalog if int(i["price_max"]) >= b_min_r and int(i["price_min"]) <= b_max_r]
            if not valid_candidates: valid_candidates = catalog

        prompt = prompt_func(q, valid_candidates, b_min, b_max, top_k_llm)
        raw_text = call_llm(token, prompt, model)
        found_ids = re.findall(r'item_\d{3}', raw_text)

        unique_recs = []
        for r in found_ids:
            if r in [c["id"] for c in valid_candidates] and r not in unique_recs:
                unique_recs.append(r)

        # Оценка релевантности (NDCG) и Serendipity
        if "Rerank" in config_name:
            scored_pool = []
            for cid in unique_recs:
                item = next(c for c in catalog if c["id"] == cid)
                tag_overlap = len(interests.intersection(set(item["tags"].split("|"))))
                C, m = 50, 4.2
                rating = float(item["rating"])
                reviews = int(item["reviews_count"])
                bayesian_rating = (C * m + reviews * rating) / (C + reviews)

                llm_score = 0
                if "Weight" in config_name:
                    match = re.search(rf'{cid}[^0-9]*(\d+)', raw_text)
                    if match: llm_score = int(match.group(1))

                if "Weight" in config_name:
                    final_score = (llm_score * 0.5) + (tag_overlap * 1.5) + (bayesian_rating * 1.0)
                else:
                    final_score = (tag_overlap * 2.0) + bayesian_rating
                scored_pool.append({"id": cid, "score": final_score})

            scored_pool.sort(key=lambda x: x["score"], reverse=True)
            final_recs = [x["id"] for x in scored_pool][:5]
        else:
            final_recs = unique_recs[:5]

        if len(final_recs) < 5:
            for cand in sorted(valid_candidates, key=lambda x: float(x["rating"]), reverse=True):
                if cand["id"] not in final_recs: final_recs.append(cand["id"])
                if len(final_recs) == 5: break

        # Считаем NDCG
        ndcg = calculate_ndcg(final_recs, q["relevant_dict"])
        total_ndcg += ndcg

        # Считаем Serendipity: есть ли в топ-5 релевантный товар без совпадения тегов
        has_serendipity = 0
        for r in final_recs:
            # Если товар релевантен (grade >= 1), но нет совпадения тегов
            if q["relevant_dict"].get(r, 0) >= 1:
                item = next(c for c in catalog if c["id"] == r)
                if len(interests.intersection(set(item["tags"].split("|")))) == 0:
                    has_serendipity = 1
                    break
        total_serendipity += has_serendipity

        results_list.append(
            {"query_id": q_id, "ndcg": ndcg, "serendipity": has_serendipity, "recs": "|".join(final_recs)})
        time.sleep(1)

    mean_ndcg = total_ndcg / len(golden_set)
    serendipity_rate = total_serendipity / len(golden_set)
    print(f"  => NDCG@5 = {mean_ndcg:.4f} | Serendipity@1 = {serendipity_rate:.4f}")

    if "H5" in config_name:
        out_file = os.path.join(BASE_DIR, "results_h5.csv")
    else:
        out_file = os.path.join(BASE_DIR, f"results_{config_name.split(':')[0].lower().replace(' ', '_')}.csv")

    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "ndcg", "serendipity", "recs"])
        writer.writeheader()
        writer.writerows(results_list)
    print(f"  => Сохранено в {out_file}")


def run_h6(golden_set, catalog, token):
    print(f"\n--- Запуск H6 (Diversified Harness 4+1) ---")
    total_ndcg = 0.0
    total_serendipity = 0
    results_list = []

    for q in golden_set:
        q_id = q["query_id"]
        b_min, b_max = int(q["budget_min"]), int(q["budget_max"])
        interests = set(q["interests"].split("|"))

        valid_candidates = []
        for item in catalog:
            p_min, p_max = int(item["price_min"]), int(item["price_max"])
            if p_max >= b_min and p_min <= b_max and q["occasion"] in item["occasions"].split("|"):
                valid_candidates.append(item)
        if len(valid_candidates) < 10:
            b_min_r, b_max_r = int(b_min * 0.7), int(b_max * 1.3)
            valid_candidates = [i for i in catalog if int(i["price_max"]) >= b_min_r and int(i["price_min"]) <= b_max_r]
            if not valid_candidates: valid_candidates = catalog

        cands_text = "\n".join([f"{c['id']} | {c['title']} | Теги: {c['tags']}" for c in valid_candidates])
        prompt = f"""Выбери 10 ID и оцени релевантность от 1 до 10.
Запрос: {q['interests']} для {q['recipient_types']}, повод: {q['occasion']}.
Список:\n{cands_text}\nВерни строго JSON-массив объектов: [{{"id": "item_XXX", "score": 9}}, ...]"""

        raw_text = call_llm(token, prompt, "GigaChat-Pro")
        found_ids = re.findall(r'item_\d{3}', raw_text)

        scored_pool = []
        for cid in found_ids:
            if cid not in [c["id"] for c in valid_candidates]: continue
            item = next(c for c in catalog if c["id"] == cid)
            tag_overlap = len(interests.intersection(set(item["tags"].split("|"))))
            C, m = 50, 4.2
            rating = float(item["rating"])
            reviews = int(item["reviews_count"])
            bayesian_rating = (C * m + reviews * rating) / (C + reviews)
            match = re.search(rf'{cid}[^0-9]*(\d+)', raw_text)
            llm_score = int(match.group(1)) if match else 0

            # Скор для очевидных (H5)
            final_score = (llm_score * 0.5) + (tag_overlap * 1.5) + (bayesian_rating * 1.0)
            scored_pool.append({"id": cid, "score": final_score, "tag_overlap": tag_overlap, "llm_score": llm_score,
                                "tags": item["tags"]})

        # Разделяем
        obvious = []
        unobvious = []
        for c in scored_pool:
            if c["tag_overlap"] > 0:
                obvious.append(c)
            else:
                unobvious.append(c)

        # Для очевидных сортируем по скору H5
        obvious.sort(key=lambda x: x["score"], reverse=True)

        # Для неочевидных сортируем по LLM-оценке (сюрприз должен быть высоко оценен нейросетью)
        unobvious.sort(key=lambda x: x["llm_score"], reverse=True)

        final_recs = [c["id"] for c in obvious[:4]]
        if unobvious:
            final_recs.append(unobvious[0]["id"])
        else:
            for c in obvious[4:]:
                if c["id"] not in final_recs: final_recs.append(c["id"])
                if len(final_recs) == 5: break

        if len(final_recs) < 5:
            for cand in sorted(valid_candidates, key=lambda x: float(x["rating"]), reverse=True):
                if cand["id"] not in final_recs: final_recs.append(cand["id"])
                if len(final_recs) == 5: break

        ndcg = calculate_ndcg(final_recs, q["relevant_dict"])
        total_ndcg += ndcg

        # Считаем Serendipity (тот же метод, что и выше)
        has_serendipity = 0
        for r in final_recs:
            if q["relevant_dict"].get(r, 0) >= 1:
                item = next(c for c in catalog if c["id"] == r)
                if len(interests.intersection(set(item["tags"].split("|")))) == 0:
                    has_serendipity = 1
                    break
        total_serendipity += has_serendipity

        results_list.append(
            {"query_id": q_id, "ndcg": ndcg, "serendipity": has_serendipity, "recs": "|".join(final_recs)})
        time.sleep(1)

    mean_ndcg = total_ndcg / len(golden_set)
    serendipity_rate = total_serendipity / len(golden_set)
    print(f"  => NDCG@5 = {mean_ndcg:.4f} | Serendipity@1 = {serendipity_rate:.4f}")

    out_file = os.path.join(BASE_DIR, "results_h6.csv")
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "ndcg", "serendipity", "recs"])
        writer.writeheader()
        writer.writerows(results_list)


if __name__ == "__main__":
    catalog, golden_set = load_catalog(), load_golden()
    token = get_token()

    run_and_save("H1_Constraint", golden_set, catalog, token, prompt_h1, "GigaChat", 5)
    run_and_save("H2_FewShot", golden_set, catalog, token, prompt_h2, "GigaChat", 5)
    run_and_save("H3_Pro", golden_set, catalog, token, prompt_h2, "GigaChat-Pro", 5)
    run_and_save("H4_MathRerank", golden_set, catalog, token, prompt_h4, "GigaChat-Pro", 10)
    run_and_save("H5_WeightedRerank", golden_set, catalog, token, prompt_h5, "GigaChat-Pro", 10)

    # Запуск H6
    run_h6(golden_set, catalog, token)

    print("\nВсе результаты сохранены в папку проекта!")
import csv
import os

import pandas as pd

CANDIDATES = "data/gold_candidates.csv"
LABELS_FILE = "gold/gold_labels.csv"
KEYS = {"1": "Bad", "2": "Neutral", "3": "Good", "s": "Skip"}
GUIDE = """Правила (оцениваем отношение автора к товару в целом):
  1 = Bad      недоволен: негатив преобладает или есть серьёзный недостаток
  3 = Good     доволен: позитив преобладает, автор хвалит или рекомендует
  2 = Neutral  смешанное мнение (плюсы и минусы примерно равны),
               «нормально, ничего особенного» или только факты без оценки
  s = пропуск, если текст непонятен или совсем не удаётся решить
Не думайте дольше 10-15 секунд: берите первое впечатление.
"""

os.makedirs("gold", exist_ok=True)
cand = pd.read_csv(CANDIDATES, dtype={"id": str})
done = set()
if os.path.exists(LABELS_FILE):
    done = set(pd.read_csv(LABELS_FILE, dtype={"id": str})["id"])
else:
    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["id", "label"])

todo = cand[~cand["id"].isin(done)]
print(f"Размечено: {len(done)}, осталось: {len(todo)}. Выход: q\n")
print(GUIDE)
for i, row in enumerate(todo.itertuples(), start=1):
    print("-" * 70)
    print(f"[{len(done) + i}/{len(cand)}]")
    print(row.text)
    while True:
        key = input("1=Bad 2=Neutral 3=Good s=пропуск q=выход > ").strip().lower()
        if key in KEYS or key == "q":
            break
    if key == "q":
        break
    with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.id, KEYS[key]])
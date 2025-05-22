
# GA Churn Predictor — CatBoost API + UI

Небольшой сервис, который:

1. Принимает **сырые** логи Google Analytics (`ga_sessions.csv` и `ga_hits.csv`);
2. Прогоняет их через **тот же** препроцессинг, что использовался при обучении;
3. Выдаёт вероятность конверсии, посчитанную готовой моделью **CatBoost**.

Работает двумя способами — через REST‑эндпойнт `/predict` **и** через удобный веб‑интерфейс `/ui`.

---

## 🗂 Структура репозитория

```
.
├─ Dockerfile                # собирает образ
├─ docker-compose.yml        # разворачивает сервис
├─ main.py                   # FastAPI + Gradio UI
├─ preprocessing_script.py   # пайплайн фичей
├─ inference.py              # загрузка CatBoost и predict
├─ models/
│   └─ catboost_model.cbm    # обученная модель
├─ samples/
│   ├─ make_samples.py       # генератор мини‑датасетов
│   ├─ sessions_sample.csv   # 20 сессий
│   ├─ hits_sample.csv       # 20 хитов
│   └─ sample_record.json    # пример одной записи
└─ README.md
```

---

## 🚀 Быстрый старт


| Что открывается | URL |
|-----------------|-----|
| Swagger (REST)  | http://127.0.0.1:8000/docs |
| Веб‑UI          | http://127.0.0.1:8000/ui |

---

## 🧪 Тест запросов

### Swagger

1. Откройте `/docs`.
2. **POST /predict** → **Try it out**.
3. Выберите `samples/sessions_sample.csv` и `samples/hits_sample.csv`.
4. **Execute** — получите JSON‑ответ.

### cURL

```bash
curl -X POST http://127.0.0.1:8000/predict      -F ga_sessions=@samples/sessions_sample.csv      -F ga_hits=@samples/hits_sample.csv
```

---

## 🖥 Веб‑интерфейс `/ui`

| Вкладка | Описание |
|---------|----------|
| **Single JSON** | Вставьте JSON одной записи (см. `samples/sample_record.json`) → вероятность. |
| **Batch CSV**   | Загрузите **два** файла (`sessions`, `hits`) → таблица с вероятностями. |

---

## FAQ

| Вопрос | Ответ |
|--------|-------|
| **Не все фичи указал** | Отсутствующие колонки добавятся автоматически. |
| **Порог классификации?** | API выдаёт вероятность; порог можно выбирать самим. |
| **Обновить модель?** | Замените `models/catboost_model.cbm` и пересоберите образ. |

Enjoy 🎉

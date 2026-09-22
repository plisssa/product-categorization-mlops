# Презентация — план слайдов и тезисы для защиты

Готовый генератор: `python docs/build_presentation.py` → `docs/presentation.pptx`
(нужен `pip install python-pptx`). Ниже — тезисы (речь) к каждому слайду.

1. **Титул.** Проект: категоризация e-commerce товаров, Production ML Spring 2026.
2. **Задача и метрика.** Предсказываем `category_ind` по тексту+картинке+URL;
   метрика macro-F1 (важен каждый класс); результат — сабмишн + сервис.
3. **EDA.** 111 405 товаров, 207 классов, дисбаланс 31 563×, картинка у 100%.
   Длинный хвост — главный ограничитель metric; vendor/url почти всегда есть.
4. **Идея и модель.** full_text → TF-IDF → E5 → E5+CLIP (fusion) → ансамбль;
   борьба с дисбалансом (class_weight, иерархия, калибровка).
5. **Воспроизводимость.** Один скрипт end2end; uv-зависимости; MLflow логирует
   параметры/метрики/модель; артефакты на S3.
6. **Результаты.** 0.288 → 0.387 (TF-IDF); далее эмбеддинги+ансамбль.
7. **Production-архитектура.** Один compose: gateway + Triton + Redis +
   Prometheus/Grafana + Evidently + Airflow.
8. **Triton/ONNX.** Ансамбль E5→голова в ONNX; конвертер prepare_models.py;
   backend local|triton (по умолчанию local — работает на CPU без GPU).
9. **Мониторинг и дрифт.** Grafana-дашборд, алерты на пороги; Evidently Data Drift.
10. **Автоматизация.** Airflow @daily (retrain+drift), GitLab CI/CD (lint/test/build).
11. **Карта критериев.** Где что реализовано (РК 15 + защита 30), опора на ДЗ 4–9.
12. **Спасибо / вопросы.**

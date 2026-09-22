# Airflow — регулярные расчёты (hw9)

DAG `ecom_retrain`: `extract -> build_features -> drift_check -> retrain -> publish_s3`,
расписание `@daily`. Закрывает критерии «регулярные расчёты» и «ML-мониторинг дрифта».

```bash
docker compose -f airflow/docker-compose.yml up -d
# открыть http://localhost:8088, включить DAG ecom_retrain
```

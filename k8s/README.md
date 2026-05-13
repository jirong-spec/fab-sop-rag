# Kubernetes Deployment Guide

## 架構

```
┌─────────────────────── K8s Cluster (fab-sop namespace) ───────────────────────┐
│                                                                                 │
│  Ingress (TLS / nginx)                                                          │
│       │                                                                         │
│  api Deployment (replicas=2, HPA 2→8)  ←── ConfigMap + Secret                  │
│       │                    │                                                    │
│       │             chroma-pvc / hf-cache-pvc (ReadWriteMany)                  │
│       │                                                                         │
│  neo4j StatefulSet ── neo4j-backup CronJob (daily 02:00)                       │
│                                                                                 │
│  vllm Deployment (GPU nodeSelector, replicas=1)                                 │
│       └── hostPath /home/jimmy/models (GPU node local SSD)                     │
│                                                                                 │
│  mlflow Deployment ── postgres StatefulSet                                      │
│       └── S3 artifact store                                                     │
│                                                                                 │
│  eval CronJob (weekly Mon 03:00 → logs to MLflow)                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 部署順序

```bash
# 1. Namespace
kubectl apply -f k8s/namespace.yaml

# 2. 設定 Secret（填入真實密碼後再 apply）
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml

# 3. 儲存層（PVC 先建，StatefulSet 才能 bind）
kubectl apply -f k8s/api/pvc.yaml
kubectl apply -f k8s/neo4j/statefulset.yaml
kubectl apply -f k8s/neo4j/service.yaml
kubectl apply -f k8s/neo4j/backup-cronjob.yaml

# 4. MLflow backend
kubectl apply -f k8s/mlflow/postgres-statefulset.yaml
kubectl apply -f k8s/mlflow/deployment.yaml
kubectl apply -f k8s/mlflow/service.yaml

# 5. vLLM（GPU node 必須已加 taint nvidia.com/gpu=true:NoSchedule）
kubectl apply -f k8s/vllm/deployment.yaml
kubectl apply -f k8s/vllm/service.yaml

# 6. API（等 neo4j + vllm ready 後）
kubectl apply -f k8s/api/deployment.yaml
kubectl apply -f k8s/api/service.yaml
kubectl apply -f k8s/api/hpa.yaml
kubectl apply -f k8s/api/ingress.yaml   # 需要 cert-manager + nginx ingress

# 7. 定期 eval
kubectl apply -f k8s/eval/cronjob.yaml

# 確認狀態
kubectl get all -n fab-sop
```

## MLflow 使用方式

```bash
# 手動觸發一次 eval 並記錄到 MLflow
docker compose exec -T api python scripts/eval_compare.py \
  --mlflow-uri http://localhost:5000

# 在 K8s 內手動觸發 CronJob
kubectl create job --from=cronjob/eval-weekly eval-manual -n fab-sop

# 查看 MLflow UI（port-forward）
kubectl port-forward svc/mlflow-svc 5000:5000 -n fab-sop
# 開啟 http://localhost:5000
```

每次 eval run 記錄：

| 類型 | 內容 |
|------|------|
| **Params** | model、reranker、cap_strategy、cot_method、eval_mode |
| **Metrics** | retrieval_rate、answer_rate、multihop_answer_rate、avg_latency_ms、各題 R/A/latency |
| **Artifacts** | prompt template（txt）、完整 eval 結果（JSON） |

## Secret 欄位說明

| Key | 說明 |
|-----|------|
| `NEO4J_PASSWORD` | Neo4j auth 密碼 |
| `API_KEY` | FastAPI `X-API-Key` |
| `MLFLOW_DB_PASSWORD` | PostgreSQL mlflow 用戶密碼 |
| `AWS_ACCESS_KEY_ID` | S3 artifact store 存取金鑰 |
| `AWS_SECRET_ACCESS_KEY` | S3 artifact store 密鑰 |

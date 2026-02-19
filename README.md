# MedGemma Rehab Bot Local Environment

This folder gives you a minimal local stack for:
- dataset preprocessing from Hugging Face
- MedGemma inference service with vLLM in Docker

## 1) Prerequisites

- Docker Engine + Docker Compose v2
- NVIDIA driver + NVIDIA Container Toolkit
- Hugging Face account/token with model access

Quick GPU sanity check:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 2) Configure env

```bash
cd medgemma_rehab_bot
cp .env.example .env
```

Edit `.env`:
- set `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`
- set `VLLM_MODEL` to the exact model ID you can access

## 3) Build rehab training data

```bash
docker compose build preprocess
docker compose run --rm preprocess
```

Output file:
- `medgemma_rehab_bot/data/rehab_train.jsonl`

## 4) Start vLLM API

```bash
docker compose up -d vllm
docker compose logs -f vllm
```

When server is ready, test:

```bash
curl http://localhost:8000/v1/models
```

Simple chat test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"'"${VLLM_MODEL}"'",
    "messages":[
      {"role":"system","content":"You are a cautious rehab assistant. Include safety warnings."},
      {"role":"user","content":"I have mild ankle pain after running. Give a home rehab plan."}
    ],
    "temperature":0.2
  }'
```

## 5) Notes for competition path

- Use this stack first for RAG baseline and reproducible demo.
- If you later fine-tune with LoRA, keep vLLM as the deployment layer.

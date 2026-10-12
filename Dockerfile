# Development-only offline fallback image. Rebase on the official image once its
# exact name/digest is published; do not submit this generic Python base as-is.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BASELINE_TRAIN_FILE=/opt/data/train.jsonl

WORKDIR /root/participant
COPY . /root/participant/
COPY data/raw/训练-验证-数据集/train/train_public.jsonl /opt/data/train.jsonl
RUN chmod +x /root/participant/start.sh

ENTRYPOINT ["/root/participant/start.sh"]


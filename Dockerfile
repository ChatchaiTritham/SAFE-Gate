FROM python:3.11.15-slim

LABEL maintainer="Chatchai Tritham <chatchait66@nu.ac.th>"
LABEL repository="SAFE-Gate"
LABEL description="Reproducible research container for SAFE-Gate"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements.lock ./
RUN pip install --upgrade pip && pip install -r requirements.lock && pip install matplotlib seaborn pyyaml tqdm

COPY . .
RUN pip install -e .

CMD ["python", "-m", "pytest", "tests", "-q"]

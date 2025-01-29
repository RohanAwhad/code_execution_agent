FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    make \
    cmake \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpq-dev \
    tzdata \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    tk-dev \
    libharfbuzz-dev \
    libgl1-mesa-glx \
    libfribidi-dev \ 
    gcc \
    g++ \
    gfortran && \
    rm -rf /var/lib/apt/lists/*

COPY base_requirements.txt requirements.txt ./
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache -r base_requirements.txt -r requirements.txt

COPY streamlit_chatbot.py .
COPY src ./src
RUN mkdir data

FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "streamlit", "run", "streamlit_chatbot.py"]
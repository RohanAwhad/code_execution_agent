FROM python:3.10-slim-bullseye

WORKDIR /app

# Install necessary build dependencies
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

COPY base_requirements.txt ./base_requirements.txt

# Upgrade pip and install packages with increased timeout
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r base_requirements.txt



COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY streamlit_chatbot.py .
RUN mkdir data

CMD ["python", "-m", "streamlit", "run", "streamlit_chatbot.py"]

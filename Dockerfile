FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV KERAS_BACKEND=tensorflow
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache

RUN mkdir -p /tmp/.streamlit && \
    echo '[server]\nheadless = true\nport = 7860\naddress = "0.0.0.0"\nenableCORS = false\nenableXsrfProtection = false\n' \
    > /tmp/.streamlit/config.toml

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip

WORKDIR /app

RUN git clone https://github.com/salsabilafauzh/streamlit-lstm.git .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "prediction.py", "--server.port=8501", "--server.address=0.0.0.0"]

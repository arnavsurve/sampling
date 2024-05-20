FROM python:3.8-slim

WORKDIR ../

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "popularity.py"]

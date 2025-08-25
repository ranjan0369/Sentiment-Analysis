FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && pip install pipenv
COPY Pipfile Pipfile.lock ./
RUN pipenv install
COPY app.py .
COPY templates/ templates/
COPY models/ models/
COPY nltk_data/ nltk_data/
EXPOSE 5000
CMD ["pipenv", "run", "python", "app.py"]
FROM python:3.11-slim

WORKDIR /app

COPY requirements_django.txt .
RUN pip install --no-cache-dir -r requirements_django.txt

COPY . .

RUN python manage.py collectstatic --noinput || true

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

FROM python:3.7-alpine

WORKDIR /flask_app

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Copy only requirements first to cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code (app.py, etc.)
COPY . .

EXPOSE 5000
CMD ["flask", "run"]
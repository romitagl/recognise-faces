FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# expose port 5555
EXPOSE 5555

# Serve the model
CMD ["python", "serve_model.py"]

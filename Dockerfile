# Use official Python image
FROM python:3.11

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose port (use 8000 or your preferred port)
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "career_ai.src.api:app", "--host", "0.0.0.0", "--port", "8000"]

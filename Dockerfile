# Use an official Python image as a base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install system dependencies
RUN apt-get update && apt-get install -y pngquant curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Expose the FastAPI default port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
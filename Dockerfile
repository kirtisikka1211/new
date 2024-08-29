# Use the official Python image with a slim base
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8505

# Start ngrok and run the Streamlit app
CMD ["python3", "run_ngrok.py"]

# Use the official Python image with a slim base
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libgthread-2.0-0

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Start the Streamlit app
CMD ["streamlit", "run", "app.py"]

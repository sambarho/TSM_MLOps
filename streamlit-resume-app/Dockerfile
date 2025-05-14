# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /streamlit-resume-app

# Copy the requirements.txt file into the container at /streamlit-resume-app
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /streamlit-resume-app
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Set the command to run Streamlit when the container starts
CMD ["streamlit", "run", "simple_app.py"]

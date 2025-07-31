# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
# Using --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Expose the ports that the two Streamlit apps will run on
EXPOSE 8501
EXPOSE 8502

# The command to run when the container launches.
# This executes the main app.py which acts as a launcher.
CMD ["python", "app.py"]
# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flask-cors

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]

FROM python:3.9

# Set the working directory
WORKDIR /app

COPY app.py requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 8080
COPY iris.csv /app/iris.csv



# Command to run the application
CMD ["python", "app.py"]

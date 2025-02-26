# Use the official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Create the 'uploads' directory and set permissions
RUN mkdir -p /app/uploads && chmod -R 777 /app/uploads

# Install required Python packages
RUN pip install -r requirements.txt

# Expose the port that the app runs on
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]

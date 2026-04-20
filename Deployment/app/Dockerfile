FROM python:3.10-slim
# Set working directory in the container
WORKDIR /app
# Copy requirements first (for caching)
COPY requirements.txt .
# Install Python dependenes
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the project into the container
COPY . .
# Expose the port api will use
EXPOSE 8000

# Set the default command to run the api
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0","--port","8000"]
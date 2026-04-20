FROM python:3.10-slim
# Set working directory in the container
WORKDIR /app
# Copy requirements first (for caching)
COPY app/requirements.txt .
# Install Python dependenes
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the project into the container
COPY app app/

# Expose the port Streamlit will use
EXPOSE 8000

# Set the default command to run the api
CMD ["uvicorn", "app.main:app --reload", "--host", "0.0.0.0","--port","8000"]
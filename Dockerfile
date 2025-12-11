FROM python:3.7.16
# Set working directory in the container
WORKDIR /app
# Copy requirements first (for caching)
COPY Deployment/requirements.txt .
# Install Python dependenes
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the project into the container
COPY . .
# Expose the port Streamlit will use
EXPOSE 8501

# Set the default command to run the Streamlit app
CMD ["streamlit", "run", "/app/Deployment/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
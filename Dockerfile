# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install pipenv
RUN pip install pipenv

# Install dependencies
RUN pipenv install --system

# Expose port 8000 (adjust as per your application needs)
EXPOSE 8000
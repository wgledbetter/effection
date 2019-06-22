FROM python:3.6

# Copy Files
COPY . /home/

# Install Python Packages
RUN pip install --trusted-host pypi.python.org -r /home/requirements.txt
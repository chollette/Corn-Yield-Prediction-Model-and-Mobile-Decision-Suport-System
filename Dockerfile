FROM python:3.6.13-slim

# Upgrade pip
RUN pip3 install --upgrade pip

## make a local directory
RUN mkdir /app

# set "app" as the working directory from which CMD, RUN, ADD references
WORKDIR /app

# now copy all the files in this directory to /code
ADD . .

# pip install the local requirements.txt
RUN pip3 install -r requirements.txt

# Define our command to be run when launching the container
CMD gunicorn app:app --bind 0.0.0.0:$PORT --reload

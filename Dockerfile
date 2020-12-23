FROM ubuntu:18.04
# Install Python 3.8 and pip
RUN apt update && \
    apt install -y software-properties-common && \
    apt-get update && \
    apt-get install -y python3.8 && \
    apt-get install -y python3.8-dev && \
    apt-get install -y python3-pip;
RUN apt-get install -y curl
RUN curl https://bootstrap.pypa.io/ez_setup.py -o - | python3.8 && python3.8 -m easy_install pip

# Set the locale
RUN apt-get clean && apt-get -y update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Project Dependencies
COPY . .
RUN pip install numpy
RUN pip install scipy
RUN pip install Cython==0.29.21
RUN pip install scikit-learn
RUN pip install -r requirements.txt
CMD ["python3.8","utils/download.py"]
CMD ["python3.8","app.py"]
EXPOSE 3355

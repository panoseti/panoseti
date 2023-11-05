FROM quay.io/jupyter/minimal-notebook

WORKDIR /home/jovyan/work/

EXPOSE 8888

COPY start.sh start.sh
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

VOLUME /home/jovyan/work/

ENTRYPOINT ["./start.sh"]
FROM ghcr.io/astral-sh/uv:python3.11-bookworm
WORKDIR /project
COPY requirements.lock.txt /project/requirements.lock.txt
RUN uv venv /opt/venv && UV_PROJECT_ENVIRONMENT=/opt/venv uv pip install -r /project/requirements.lock.txt
COPY . /project
ENV PYTHONPATH=/project/src
CMD ["bash"]

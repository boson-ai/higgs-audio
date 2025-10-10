ARG PYTORCH_VERSION=25.02-py3

FROM nvcr.io/nvidia/pytorch:$PYTORCH_VERSION

WORKDIR /workspace/higgs-audio
COPY . .

RUN python3 -m venv --prompt higgs_audio_env .venv
RUN source .venv/bin/activate && \
    pip install -r requirements.txt && \
    pip install -e . && \
    deactivate

CMD ["/bin/sh", "-c", "bash"]

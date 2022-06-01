FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install apt-mirror-updater && \
    python3 -m pip cache purge

RUN rm /etc/apt/sources.list.d/*

RUN apt-mirror-updater --auto-change-mirror

RUN conda install -y pybind11

RUN apt update && apt install -y --no-install-recommends \
	cmake \
	ninja-build \
    ssh

RUN echo "Port 22" >> /etc/ssh/ssh_config \
 && echo "PubkeyAuthentication yes" >> /etc/ssh/ssh_config

COPY authorized_keys /root/.ssh/

COPY start.sh /scripts/start.sh
RUN ["chmod", "+x", "/scripts/start.sh"]
ENTRYPOINT ["/scripts/start.sh"]

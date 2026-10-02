FROM osrf/ros:jazzy-desktop-full

RUN apt-get update && \
  apt-get install -y \
  ca-certificates \
  wget \
  python3-venv \
  qtwayland5 \ 
  x11-apps \ 
  mesa-utils

RUN wget -O /tmp/foxglove-studio.deb \
  https://get.foxglove.dev/desktop/latest/foxglove-studio-latest-linux-amd64.deb && \
  apt-get install -y /tmp/foxglove-studio.deb 

RUN apt-get update && \ 
  apt-get install -y \ 
  foxglove-studio

# --- Language server for editor-side diagnostics ---
# clangd runs inside the container (see the docker exec cmd in nvim) so it sees
# the real ROS include paths. The cache dir must exist and be owned by ubuntu
# *before* the compose file mounts a named volume over it, otherwise Docker
# seeds the volume root-owned and uid 1000 can never write its index.
RUN apt-get update && \
  apt-get install -y \
  clangd && \
  mkdir -p /home/ubuntu/.cache/clangd && \
  chown -R ubuntu:ubuntu /home/ubuntu/.cache

# Clean up apt cache to keep the image smaller
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Set up passwordless sudo for ubuntu ---
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu && \
  chmod 0440 /etc/sudoers.d/ubuntu

USER ubuntu

# RUN python3 -m venv /opt/myvenv && \
#  source /opt/myvenv/bin/activate

CMD ["bash"]

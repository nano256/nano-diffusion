# Use NGC PyTorch image, optimized CUDA and torch installation
FROM nvcr.io/nvidia/pytorch:25.06-py3

# Set the cache directory in /workspace because this is where Runpod mounts 
# the persistent volume
ENV TRANSFORMERS_CACHE=/workspace/transformers
ENV DIFFUSERS_CACHE=/workspace/diffusers

# Install git and SSH server for Runpod shell access
# Install tmux for persistent shells
RUN apt-get update && apt-get install -y \
    git \
    openssh-server \
    tmux \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH
RUN mkdir /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Create a user for better shell experience
RUN useradd -m -s /bin/bash user \
    && echo 'user:runpod' | chpasswd \
    && usermod -aG sudo user \
    && echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set up shell environment
RUN echo 'export PATH="/workspace/venv/bin:$PATH"' >> /home/user/.bashrc \
    && echo 'cd /workspace/nano-diffusion' >> /home/user/.bashrc \
    && echo 'source /workspace/venv/bin/activate 2>/dev/null || true' >> /home/user/.bashrc

# Refresh shell so that PATH gets updated
RUN bash

WORKDIR /

# Copy scripts into container
COPY setup.sh /setup.sh
COPY config.sh /config.sh

# Make setup script executable
RUN chmod +x /setup.sh

# Execute setup script at runtime because it does things in the persistent 
# volume and relies on secrets provided through the env
ENTRYPOINT ["/setup.sh"]
# Start SSH service and keep container running for shell access
CMD ["/bin/bash", "-c", "service ssh start && while true; do sleep 30; done"]
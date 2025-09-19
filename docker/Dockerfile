# Use NGC PyTorch image, optimized CUDA and torch installation
FROM nvcr.io/nvidia/pytorch:25.06-py3

# Set the cache directory in /workspace because this is where Runpod mounts 
# the persistent volume
ENV TRANSFORMERS_CACHE=/workplace/transformers
ENV DIFFUSERS_CACHE=/workplace/diffusers

# Install git (if not already present) and a process manager (supervisor). 
# Needed if started in dev mode to keep Jupyter alive and the container running 
# Nodejs is needed to run AI Toolkit
RUN apt-get update && apt-get install -y git supervisor nodejs npm gettext \
    && rm -rf /var/lib/apt/lists/*

# Supervisor conf contains the services that it has to manage and keep alive
COPY supervisord.conf.template /supervisord.conf.template

# Install Rust (needed for ruff)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

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
# If not given another task, start supervisor and run the setup
CMD ["/usr/bin/supervisord"]
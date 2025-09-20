# Use NGC PyTorch image, optimized CUDA and torch installation
FROM nvcr.io/nvidia/pytorch:25.06-py3

# Set the cache directory in /workspace because this is where Runpod mounts 
# the persistent volume
ENV TRANSFORMERS_CACHE=/workplace/transformers
ENV DIFFUSERS_CACHE=/workplace/diffusers

# Install git (if not already present)
RUN apt-get update && apt-get install -y git

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
# If no custom command is given, keep the container running so that we later 
# can connect via bash and execute the desired scripts manually.
CMD ["/bin/bash", "-c", "while true; do sleep 30; done"]
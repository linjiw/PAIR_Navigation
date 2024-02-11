#!/bin/bash

# Function to get the Ubuntu version
get_ubuntu_version() {
    lsb_release -ds
}

# Function to get the NVIDIA driver version
get_nvidia_driver_version() {
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
}

# Function to get the CUDA version
get_cuda_version() {
    cuda_version=$(nvcc --version | grep -oP "(?<=release )[\d\.]+")
    if [ -z "$cuda_version" ]; then
        echo "CUDA version not found."
    else
        echo "$cuda_version"
    fi
}

# Generate Dockerfile
generate_dockerfile() {
    cat <<EOF > Dockerfile
# Use the Ubuntu $(get_ubuntu_version) as the base image
FROM nvidia/cuda:$(get_cuda_version)-base-$(lsb_release -cs)

# Install NVIDIA driver support
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-driver-$(get_nvidia_driver_version) \
    && rm -rf /var/lib/apt/lists/*

# Install any other dependencies needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    <OTHER_DEPENDENCIES> \
    && rm -rf /var/lib/apt/lists/*

# Install iaacgym (assuming it's available via pip)
RUN pip install iaacgym

# Set the working directory inside the container
WORKDIR /app

# Copy your code and any other necessary files into the container
COPY . .

# Define the command to run your application
CMD ["python", "your_script.py"]
EOF
}

# Output versions
echo "Ubuntu version: $(get_ubuntu_version)"
echo "NVIDIA driver version: $(get_nvidia_driver_version)"
echo "CUDA version: $(get_cuda_version)"

# Generate Dockerfile
generate_dockerfile

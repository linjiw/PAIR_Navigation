# Use the Ubuntu Ubuntu 20.04.6 LTS as the base image
FROM nvidia/cuda:10.1-base-focal

# Install NVIDIA driver support
RUN apt-get update && apt-get install -y --no-install-recommends     nvidia-driver-525.147.05     && rm -rf /var/lib/apt/lists/*

# Install any other dependencies needed
RUN apt-get update && apt-get install -y --no-install-recommends     <OTHER_DEPENDENCIES>     && rm -rf /var/lib/apt/lists/*

# Install iaacgym (assuming it's available via pip)
RUN pip install iaacgym

# Set the working directory inside the container
WORKDIR /app

# Copy your code and any other necessary files into the container
COPY . .

# Define the command to run your application
CMD ["python", "your_script.py"]

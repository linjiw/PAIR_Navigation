FROM nvcr.io/nvidia/pytorch:21.09-py3
ENV DEBIAN_FRONTEND=noninteractive 

# dependencies for gym
#
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt/isaacgym

RUN useradd --create-home gymuser
USER gymuser

# copy gym repo to docker
COPY --chown=gymuser . .

# install gym modules
ENV PATH="/home/gymuser/.local/bin:$PATH"
RUN cd python && pip install -q -e .

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all


# install isaacgymenvs
# USER root
# RUN git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git /opt/IsaacGymEnvs
# RUN chown -R gymuser:gymuser /opt/IsaacGymEnvs
# USER gymuser
# RUN cd /opt/IsaacGymEnvs && pip install -e .
# WORKDIR /opt/IsaacGymEnvs
# good!


USER root
RUN git clone https://github.com/linjiw/PAIR_Navigation /opt/PAIR_Navigation
RUN chown -R gymuser:gymuser /opt/PAIR_Navigation
USER gymuser
RUN cd /opt/PAIR_Navigation && pip install -e .
COPY docker/requirements.txt /opt/PAIR_Navigation
# switch to branch
RUN cd /opt/PAIR_Navigation && git checkout docker
RUN pip install -r /opt/PAIR_Navigation/requirements.txt
ENV WANDB_API_KEY=f28f905cf0d1b2c32ca1a1e437fb871c2b0e14c2
WORKDIR /opt/PAIR_Navigation/isaacgymenvs
# Run the training script
CMD ["python", "train.py"]

# good!
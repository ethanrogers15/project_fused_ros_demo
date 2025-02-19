FROM ros:humble-ros-base AS root

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Turn off interactivity
ENV DEBIAN_FRONTEND=noninteractive

# Set up working directory
WORKDIR /project_fused_ros_demo

# Install useful tools/dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    git-lfs \
    ssh \
    curl \
    nano \
    vim \
    less \
    usbutils \
    protobuf-compiler \
    autoconf \
    libtool \
    rsync \
    libboost-all-dev \
    openssh-client \
    libgl1-mesa-glx \
    x11-apps \
    v4l-utils \
    kmod

# Install python tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-opencv

# Install ROS tools
RUN apt-get update && apt-get install -y \
    ros-humble-rviz-common \
    ros-humble-rviz2 \
    ros-humble-usb-cam

# Use pip for other packages
RUN pip install --upgrade pip && pip install pydantic ipython matplotlib "numpy<2.0" mediapipe

# Copy src directory
COPY src src

# Install ROS2 dependencies using rosdep
RUN apt-get update && sudo rosdep fix-permissions && rosdep update && rosdep install --from-paths /project_fused_ros_demo/src --ignore-src -r -y

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set entrypoint script
ENTRYPOINT ["/project_fused_ros_demo/ros_ws_entrypoint.sh"]

# Add docker user with same UID and GID as your host system
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
FROM root AS image-nonroot
ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch from root to user
USER $USERNAME

# Source the ROS setup file
# Create or touch .bashrc and add ROS setup commands for non-root user
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc \
    && echo "[[ -f /project_fused_ros_demo/install/setup.bash ]] && source /project_fused_ros_demo/install/setup.bash" >> ~/.bashrc

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME
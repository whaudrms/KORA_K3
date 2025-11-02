FROM osrf/ros:noetic-desktop

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Seoul
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# 기본 유틸 + X11 + SSH "클라이언트"
RUN apt-get update && apt-get install -y \
    tzdata locales curl gnupg2 lsb-release ca-certificates \
    iproute2 net-tools iputils-ping netcat \
    openssh-client wget vim less git nano pip \
    mesa-utils libgl1-mesa-glx libglu1-mesa libceres-dev \
    libx11-6 libxext6 libxrender1 libxrandr2 libxi6 libxinerama1 \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# Gazebo가 필요하면 설치(선택)
RUN apt-get update && apt-get install -y \
    gazebo11 \
    ros-noetic-gazebo-ros ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control \
    ros-noetic-gazebo-plugins ros-noetic-joint-state-publisher-gui \
    ros-noetic-sparse-bundle-adjustment \ 
    ros-noetic-map-server ros-noetic-map-server \
    ros-noetic-geographic-msgs ros-noetic-ackermann-msgs \
    ros-noetic-amcl \ 
    libgeographic-dev python3-tk python3-shapely \   
 && apt-get clean && rm -rf /var/lib/apt/lists/*
 
RUN python3 -m pip install --user networkx scikit-image scipy pandas 

# ROS1 자동 로드(1회만)
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

WORKDIR /root


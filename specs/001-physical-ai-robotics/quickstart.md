# Quickstart Guide: Physical AI & Humanoid Robotics Book

## Prerequisites

- Ubuntu 22.04 LTS (recommended) or compatible Linux distribution
- Basic programming knowledge (Python/C++)
- Fundamental understanding of AI/ML concepts
- Minimum 16GB RAM, recommended 64GB
- NVIDIA GPU with CUDA support (minimum RTX 3060, recommended RTX 4090)

## Environment Setup

### 1. Install ROS 2 Humble Hawksbill

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep2
sudo rosdep init
rosdep update

# Source ROS 2
source /opt/ros/humble/setup.bash
```

### 2. Install Gazebo (Fortress)

```bash
# Add Gazebo repository
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-gazebo-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-gazebo-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Install Gazebo Fortress
sudo apt update
sudo apt install gazebo-fortress
```

### 3. Install NVIDIA Isaac ROS

```bash
# Install Isaac ROS dependencies
sudo apt install nvidia-isaac-ros-gxf-components
sudo apt install nvidia-isaac-ros-dev-tools
sudo apt install nvidia-isaac-ros-gxf-isaac-message-bridge
```

### 4. Set up Docusaurus for Book Development

```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs

# Create Docusaurus project
npx create-docusaurus@latest website-name classic
cd website-name
npm install
```

## First Steps

### 1. Create Your First ROS 2 Package

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build

# Source the workspace
source install/setup.bash

# Create a new package
cd src
ros2 pkg create --build-type ament_python my_robot_controller
```

### 2. Run a Basic Gazebo Simulation

```bash
# Launch Gazebo
gazebo --verbose

# Or launch with a specific world
gazebo -s libgazebo_ros_init.so -s libgazebo_ros_factory.so worlds/empty.world
```

### 3. Test Isaac Perception Pipeline

```bash
# Run a simple perception node
ros2 launch nvidia_isaac_perceptor launch/perceptor.launch.py
```

## Book Structure

The book is organized into 4 modules over 13 weeks:

1. **Module 1 (Weeks 1-3)**: The Robotic Nervous System (ROS 2)
2. **Module 2 (Weeks 4-6)**: The Digital Twin (Gazebo & Unity)
3. **Module 3 (Weeks 7-9)**: The AI-Robot Brain (NVIDIA Isaac)
4. **Module 4 (Weeks 10-11)**: Vision-Language-Action (VLA)
5. **Capstone (Weeks 12-13)**: The Autonomous Humanoid

## Getting Help

- ROS 2 Documentation: https://docs.ros.org/
- Gazebo Documentation: https://gazebosim.org/docs/
- Isaac ROS Documentation: https://nvidia-isaac-ros.github.io/
- Book Repository: [Repository link will be available after setup]
---
sidebar_position: 1
---

# Hardware Specifications & Setup Guide

This appendix provides detailed hardware specifications and setup guidelines for the Physical AI & Humanoid Robotics book. The hardware recommendations are designed to support both simulation-based learning and physical implementation.

## System Requirements

### Minimum Specifications
- **Operating System**: Ubuntu 22.04 LTS (recommended) or compatible Linux distribution
- **CPU**: Multi-core processor (Intel i7 or equivalent AMD Ryzen)
- **RAM**: 16 GB (32 GB recommended for simulation)
- **Storage**: 500 GB SSD (1 TB recommended for development)
- **Graphics**: NVIDIA GPU with CUDA support (minimum RTX 3060)

### Recommended Specifications
- **Operating System**: Ubuntu 22.04 LTS
- **CPU**: Intel i9 or AMD Ryzen 9 with 16+ cores
- **RAM**: 64 GB DDR4/DDR5
- **Storage**: 1 TB+ NVMe SSD
- **Graphics**: NVIDIA RTX 4090 or RTX 6000 Ada for advanced simulation
- **Network**: Gigabit Ethernet, 802.11ac WiFi

## Workstation Configuration

### RTX Simulation Workstation
The recommended workstation for simulation and development:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RTX Simulation Workstation                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  ROS 2 Humble   │  │   Gazebo        │  │  NVIDIA Isaac   │  │
│  │  (Middleware)   │  │   Fortress      │  │     Sim         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**
- **CPU**: AMD Ryzen Threadripper PRO 5975WX (32-core, 64-thread)
- **GPU**: NVIDIA RTX 6000 Ada Generation (48 GB VRAM)
- **RAM**: 128 GB DDR4 ECC
- **Storage**: 2 TB NVMe + 8 TB storage array
- **Cooling**: Liquid cooling system for sustained performance

### Cost Breakdown (Estimated)
- CPU: $2,500
- GPU: $6,800
- RAM: $1,000
- Motherboard: $600
- Storage: $800
- Power Supply & Cooling: $600
- **Total**: ~$12,300

## Edge AI Platforms

### NVIDIA Jetson AGX Orin 64GB
Recommended for edge deployment of AI models:

**Specifications:**
- **AI Performance**: 275 TOPS (INT8)
- **GPU**: 2048-core NVIDIA Ampere architecture GPU
- **CPU**: 12-core ARM v8.2 64-bit CPU
- **Memory**: 64 GB LPDDR5
- **Power**: Up to 60W
- **Connectivity**: PCIe Gen4 x8, 10GBASE-T Ethernet

**Use Cases:**
- Vision-Language-Action model execution
- Real-time perception and planning
- Autonomous navigation
- Humanoid robot control

**Cost**: ~$1,199

### Alternative Edge Platforms

#### Jetson Orin NX (32GB)
- **AI Performance**: 100 TOPS (INT8)
- **Memory**: 32 GB LPDDR5
- **Cost**: ~$499
- **Best for**: Budget-conscious implementations

#### Jetson Orin Nano (8GB)
- **AI Performance**: 40 TOPS (INT8)
- **Memory**: 8 GB LPDDR5
- **Cost**: ~$99
- **Best for**: Educational and prototyping

## Robotic Platforms

### Entry Level: TurtleBot 3

#### Waffle Pi
**Specifications:**
- **Processor**: Raspberry Pi 3 Model B+
- **Sensors**: LDS-01 Lidar, IMU, Camera
- **Actuators**: 2 DYNAMIXEL servos
- **Dimensions**: 345 x 301 x 128 mm
- **Weight**: 2.4 kg

**Cost**: ~$650

#### Burger
**Specifications:**
- **Processor**: Raspberry Pi 3 Model B+
- **Sensors**: LDS-01 Lidar, IMU
- **Actuators**: 2 DYNAMIXEL servos
- **Dimensions**: 190 x 140 x 220 mm
- **Weight**: 1.4 kg

**Cost**: ~$450

### Advanced: Unitree Robots

#### Go1 Quadruped
**Specifications:**
- **Motors**: 12 high-performance servo motors
- **Sensors**: IMU, stereo camera, LIDAR
- **AI**: NVIDIA Jetson Xavier NX
- **Speed**: 3 m/s
- **Battery**: 2.5 hours

**Cost**: ~$20,000

#### H1 Humanoid
**Specifications:**
- **DOF**: 22 degrees of freedom
- **Height**: 1.45 m
- **Weight**: 47 kg
- **Sensors**: Stereo cameras, IMU, force/torque sensors
- **AI**: Integrated NVIDIA Jetson platform

**Cost**: ~$70,000

## Sensor Specifications

### RGB-D Cameras
- **Intel RealSense D435**: Depth + RGB, 90° FOV
- **Intel RealSense D455**: Upgraded version with better accuracy
- **Orbbec Astra Pro**: Alternative RGB-D solution
- **ZED 2i**: Stereo camera with depth sensing

### LIDAR Sensors
- **Hokuyo UAM-05LP**: 2D LIDAR, 5m range, 10Hz
- **SICK TIM571**: 2D LIDAR, 10m range, 15Hz
- **Velodyne Puck**: 3D LIDAR, 100m range, 10Hz
- **Ouster OS0**: 3D LIDAR, 75m range, 10Hz

### IMU Sensors
- **Bosch BNO055**: 9-axis absolute orientation sensor
- **SparkFun IMU Breakout**: ICM-20948 9-axis
- **Adafruit BNO055**: Absolute orientation sensor
- **MTi-30**: High-precision IMU (Xsens)

## Simulation Hardware Requirements

### Minimum for Gazebo Simulation
- **GPU**: NVIDIA GTX 1060 or equivalent
- **VRAM**: 6 GB minimum
- **RAM**: 16 GB
- **CPU**: 4+ cores with good single-thread performance

### Recommended for Isaac Sim
- **GPU**: NVIDIA RTX 3080 or better
- **VRAM**: 10+ GB
- **RAM**: 32+ GB
- **CPU**: 8+ cores
- **Storage**: Fast SSD for asset loading

## Setup Procedures

### Ubuntu 22.04 LTS Installation
1. Download Ubuntu 22.04 LTS ISO
2. Create bootable USB drive
3. Install with appropriate partitioning for development
4. Install NVIDIA drivers for CUDA support
5. Configure user accounts and security

### ROS 2 Humble Setup
1. Add ROS 2 repository and keys
2. Install ROS 2 Humble packages
3. Configure ROS 2 environment
4. Install additional tools (RViz, rqt, etc.)

### Isaac ROS Setup
1. Install NVIDIA drivers and CUDA
2. Install Isaac ROS packages
3. Configure GPU access
4. Test perception pipelines

## Safety Considerations

### Physical Safety
- Maintain safe distances during robot operation
- Use safety barriers when appropriate
- Ensure emergency stop procedures
- Regular safety inspections

### Electrical Safety
- Proper grounding of all equipment
- Use appropriate power supplies
- Follow electrical safety standards
- Regular equipment maintenance

## Cost Analysis

### Budget Implementation
- Workstation: $2,000-4,000
- Entry robot: $450-650
- Sensors: $200-500
- **Total**: $2,650-5,150

### Professional Implementation
- Workstation: $8,000-15,000
- Advanced robot: $20,000-70,000
- Professional sensors: $1,000-5,000
- **Total**: $29,000-90,000

## Procurement Recommendations

### Academic Institutions
- Consider bulk purchasing for multiple units
- Evaluate academic discounts
- Plan for 3-5 year refresh cycles
- Budget for maintenance and support

### Individual Researchers
- Start with simulation-focused setup
- Gradually add hardware components
- Consider shared resources with colleagues
- Plan for upgrade paths

## References
1. NVIDIA Corporation. (2023). *Jetson AGX Orin Developer Kit Hardware User Guide*. https://developer.nvidia.com/
2. Open Robotics. (2023). *ROS 2 Installation Guide*. https://docs.ros.org/
3. Robotis. (2023). *TurtleBot 3 e-Manual*. https://emanual.robotis.com/
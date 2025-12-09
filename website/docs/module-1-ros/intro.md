---
sidebar_position: 1
---

# Module 1: The Robotic Nervous System (ROS 2)

## Overview

Module 1 focuses on Robot Operating System 2 (ROS 2), which serves as the foundational middleware for all robotic applications. Like the nervous system in biological organisms, ROS 2 enables communication between sensors, processors, and actuators in robotic systems.

## Learning Objectives

By the end of this module, you will be able to:
- Explain the architecture and communication patterns of ROS 2
- Create ROS 2 packages for basic robotic applications
- Implement publisher-subscriber communication patterns
- Use services and actions for synchronous and goal-oriented communication
- Configure TF transforms for coordinate frame management

## Module Structure

This module spans 3 weeks of content:
- **Week 1**: ROS 2 Architecture & Communication Patterns (800-1000 words)
- **Week 2**: Building ROS 2 Packages & Nodes (800-1000 words)
- **Week 3**: TF, Transforms & Robot State Management (600-800 words)

## Key Architecture

### ROS 2 Communication Architecture

```
┌─────────────────┐    Publish    ┌─────────────────┐
│   Sensor Node   │ ─────────────▶│     Topic       │
│                 │               │                 │
│  (Publisher)    │               │   (Message Bus) │
└─────────────────┘               └─────────────────┘
                                          │
                                          │ Subscribe
                                          ▼
                              ┌─────────────────────────┐
                              │   Processing Node       │
                              │                         │
                              │   (Subscriber)          │
                              └─────────────────────────┘
```

### Node Communication Patterns

ROS 2 supports three primary communication patterns:

1. **Topics (Publish/Subscribe)**: Asynchronous, many-to-many communication
2. **Services (Request/Response)**: Synchronous, one-to-one communication
3. **Actions (Goal/Feedback/Result)**: Asynchronous, goal-oriented communication

### TF Transform System Architecture

```
World Frame (map)
       │
       ▼
  Robot Base (base_link)
       │
       ├─► Sensor Frame (camera_link)
       │
       ├─► End Effector (gripper_link)
       │
       └─► Wheel Frames (wheel_fl_link, etc.)
```

## Technical Implementation

### Required Components

- **ROS 2 Humble Hawksbill**: Long-term support distribution
- **rclcpp/rclpy**: Client libraries for C++/Python
- **DDS Middleware**: Data Distribution Service implementation
- **RViz2**: Visualization tool for debugging
- **ros2 bag**: Data recording and playback

### Quality of Service (QoS) Settings

Different communication scenarios require different QoS profiles:

- **Reliable**: For critical messages that must be delivered
- **Best Effort**: For high-frequency data where occasional loss is acceptable
- **Transient Local**: For messages that new subscribers should receive
- **Volatile**: For messages that only current subscribers need

## Integration with Other Modules

ROS 2 serves as the foundational communication layer for all subsequent modules:
- **Module 2**: Gazebo simulation nodes communicate via ROS 2
- **Module 3**: Isaac ROS components integrate through ROS 2 interfaces
- **Module 4**: VLA models communicate with robotic systems through ROS 2

## Assessment Criteria

Students will demonstrate competency through:
- Creation of a ROS 2 package that implements publisher-subscriber communication
- Implementation of a service server for robot control
- Configuration of TF transforms for a simple robot model
- Integration of simulated sensors with ROS 2 communication

## References

1. Open Robotics. (2023). *ROS 2 Humble Hawksbill Documentation*. https://docs.ros.org/en/humble/
2. Quigley, M., et al. (2009). ROS: an open-source Robot Operating System. ICRA Workshop on Open Source Software.
3. Macenski, S., et al. (2022). ROS 2 Design: The Next Generation of the Robot Operating System. arXiv preprint arXiv:2201.01890.
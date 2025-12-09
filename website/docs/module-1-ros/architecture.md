---
sidebar_position: 2
---

# ROS 2 Architecture & Communication Patterns

## Overview

This chapter explores the architecture of Robot Operating System 2 (ROS 2) and its communication patterns that form the foundation of the robotic nervous system. Understanding ROS 2 architecture is crucial for building robust and scalable robotic applications.

## Learning Objectives

By the end of this chapter, you will be able to:
- Describe the core architecture of ROS 2
- Implement publisher-subscriber communication patterns
- Use services for synchronous communication
- Apply actions for goal-oriented communication
- Configure Quality of Service (QoS) settings for different scenarios

## ROS 2 Architecture Overview

### Core Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ROS 2 Architecture                       │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Client    │  │    DDS      │  │  Middleware         │  │
│  │  Libraries  │  │  (RMW)      │  │  Implementation     │  │
│  │ (rclcpp/    │  │             │  │  (Fast DDS,         │  │
│  │  rclpy)     │  │             │  │   Cyclone DDS, etc.)│  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Nodes &       │
                    │ Communication   │
                    └─────────────────┘
```

### Client Library Architecture

ROS 2 uses a layered architecture with client libraries:

- **Application Layer**: User code in C++, Python, etc.
- **Client Library Layer**: rclcpp, rclpy, rclrs
- **ROS Middleware (RCL)**: Common interface layer
- **Middleware Interface (RMW)**: DDS abstraction layer
- **DDS Implementation**: Data Distribution Service

## Communication Patterns

### 1. Publisher-Subscriber (Topics)

The most common communication pattern for streaming data:

```cpp
// C++ Publisher Example
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher() : Node("minimal_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "Hello, World! " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
};
```

```
┌─────────────────┐    Publish    ┌─────────────────┐
│   Publisher     │ ─────────────▶│     Topic       │
│   (Node A)      │               │                 │
└─────────────────┘               │   (Message Bus) │
        │                         │                 │
        │ Subscribe               └─────────────────┘
        ▼                                  │
┌─────────────────┐                        │
│   Subscriber    │◀───────────────────────┘
│   (Node B)      │
└─────────────────┘
```

### 2. Request-Response (Services)

For synchronous, one-to-one communication:

```python
# Python Service Server Example
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

### 3. Goal-Oriented (Actions)

For long-running tasks with feedback:

```cpp
// C++ Action Server Example
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "example_interfaces/action/fibonacci.hpp"

class MinimalActionServer : public rclcpp::Node
{
public:
  using Fibonacci = example_interfaces::action::Fibonacci;
  using GoalHandleFibonacci = rclcpp_action::ServerGoalHandle<Fibonacci>;

  explicit MinimalActionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("minimal_action_server", options)
  {
    using namespace std::placeholders;

    this->action_server_ = rclcpp_action::create_server<Fibonacci>(
      this,
      "fibonacci",
      std::bind(&MinimalActionServer::handle_goal, this, _1, _2),
      std::bind(&MinimalActionServer::handle_cancel, this, _1),
      std::bind(&MinimalActionServer::handle_accepted, this, _1));
  }

private:
  rclcpp_action::Server<Fibonacci>::SharedPtr action_server_;
};
```

## Quality of Service (QoS) Profiles

QoS settings control communication behavior:

### Reliability Policy
- **RELIABLE**: All messages are delivered (with retries)
- **BEST_EFFORT**: Messages may be lost but faster delivery

### Durability Policy
- **TRANSIENT_LOCAL**: Late-joining subscribers get last message
- **VOLATILE**: Only future messages delivered to new subscribers

### History Policy
- **KEEP_LAST**: Maintain N most recent messages
- **KEEP_ALL**: Maintain all messages (limited by resource limits)

### Example QoS Configuration

```cpp
// Reliable communication for critical data
rclcpp::QoS reliable_qos(10);
reliable_qos.reliable().transient_local().keep_last(10);

// Best effort for high-frequency sensor data
rclcpp::QoS best_effort_qos(10);
best_effort_qos.best_effort().keep_last(10);
```

## ROS 2 Launch System

### Launch Files

Launch files enable starting multiple nodes with configuration:

```python
# launch/my_robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_pkg',
            executable='publisher_node',
            name='publisher',
            parameters=[
                {'param1': 'value1'},
                {'param2': 42}
            ],
            remappings=[
                ('original_topic', 'remapped_topic')
            ]
        ),
        Node(
            package='my_robot_pkg',
            executable='subscriber_node',
            name='subscriber'
        )
    ])
```

## Parameter System

ROS 2 provides a distributed parameter system:

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with defaults
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_velocity', 1.0)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value

        # Parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.DOUBLE:
                self.max_velocity = param.value
        return SetParametersResult(successful=True)
```

## TF (Transform) System

The Transform (TF) system manages coordinate frame relationships:

```cpp
// Transform broadcaster
#include "tf2_ros/transform_broadcaster.h"

class FramePublisher : public rclcpp::Node
{
public:
  FramePublisher() : Node("frame_publisher")
  {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    timer_ = this->create_wall_timer(
      50ms, std::bind(&FramePublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "base_link";
    t.child_frame_id = "laser_frame";
    t.transform.translation.x = 0.1;
    t.transform.translation.y = 0.0;
    t.transform.translation.z = 0.2;
    t.transform.rotation.x = 0.0;
    t.transform.rotation.y = 0.0;
    t.transform.rotation.z = 0.0;
    t.transform.rotation.w = 1.0;
    tf_broadcaster_->sendTransform(t);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
```

## Best Practices

### Node Design
- Keep nodes focused on single responsibilities
- Use parameters for configuration
- Implement proper error handling
- Follow naming conventions

### Communication Design
- Choose appropriate communication pattern for use case
- Use appropriate QoS settings for data type
- Consider message size and frequency
- Plan for network partitioning

### Performance Considerations
- Use intra-process communication when possible
- Optimize message sizes
- Consider data serialization overhead
- Monitor network usage

## Integration with Physical AI

ROS 2 serves as the foundational communication layer for all Physical AI modules:
- **Simulation**: Gazebo nodes communicate via ROS 2
- **Perception**: Isaac ROS components use ROS 2 interfaces
- **Control**: VLA systems integrate through ROS 2 messaging
- **Hardware**: Real robot drivers use ROS 2 protocols

## References

1. Open Robotics. (2023). *ROS 2 Design: The Next Generation of the Robot Operating System*. https://docs.ros.org/en/humble/
2. Quigley, M., et al. (2009). ROS: an open-source Robot Operating System. ICRA Workshop on Open Source Software.
3. Macenski, S., et al. (2022). ROS 2 Design: The Next Generation of the Robot Operating System. arXiv preprint arXiv:2201.01890.
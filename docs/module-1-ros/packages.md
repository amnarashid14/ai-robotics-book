---
sidebar_position: 3
---

# Building ROS 2 Packages & Nodes

## Overview

This chapter covers the creation and structure of ROS 2 packages, which are the fundamental building blocks for organizing robotic software. Understanding package structure and node development is essential for implementing the robotic nervous system.

## Learning Objectives

By the end of this chapter, you will be able to:
- Create ROS 2 packages using standard templates
- Structure packages according to best practices
- Implement nodes in both C++ and Python
- Configure package dependencies and build systems
- Package and distribute ROS 2 software

## Package Structure

### Standard Package Layout

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata and dependencies
├── src/                    # Source code files
│   ├── publisher_node.cpp
│   └── subscriber_node.cpp
├── include/                # Header files (C++)
│   └── my_robot_package/
│       └── my_header.hpp
├── scripts/                # Executable scripts (Python, etc.)
├── launch/                 # Launch files
│   └── my_robot.launch.py
├── config/                 # Configuration files
├── test/                   # Unit tests
└── README.md               # Package documentation
```

### Package Metadata (package.xml)

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example robot package for educational purposes</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Creating Packages

### Using colcon

The standard tool for building ROS 2 packages:

```bash
# Create a new package (C++)
ros2 pkg create --build-type ament_cmake my_robot_package --dependencies rclcpp std_msgs

# Create a new package (Python)
ros2 pkg create --build-type ament_python my_robot_package_py --dependencies rclpy std_msgs
```

### C++ Package Build Configuration (CMakeLists.txt)

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Add executable
add_executable(publisher_node src/publisher_node.cpp)
ament_target_dependencies(publisher_node rclcpp std_msgs)

add_executable(subscriber_node src/subscriber_node.cpp)
ament_target_dependencies(subscriber_node rclcpp std_msgs)

# Install targets
install(TARGETS
  publisher_node
  subscriber_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## Node Implementation Examples

### C++ Node Example

```cpp
// src/my_robot_node.cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/twist.hpp"

class MyRobotNode : public rclcpp::Node
{
public:
  MyRobotNode() : Node("my_robot_node")
  {
    // Create publisher
    cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "cmd_vel", 10);

    // Create subscriber
    laser_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", 10,
      std::bind(&MyRobotNode::laser_callback, this, std::placeholders::_1));

    // Create timer
    timer_ = this->create_wall_timer(
      50ms, std::bind(&MyRobotNode::control_loop, this));

    RCLCPP_INFO(this->get_logger(), "My Robot Node initialized");
  }

private:
  void laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    // Process laser scan data
    if (!msg->ranges.empty()) {
      float min_distance = *std::min_element(msg->ranges.begin(), msg->ranges.end());
      if (min_distance < 1.0) {
        // Obstacle detected, stop robot
        stop_robot();
      }
    }
  }

  void control_loop()
  {
    // Main control logic
    auto cmd_msg = geometry_msgs::msg::Twist();
    cmd_msg.linear.x = 0.5;  // Move forward at 0.5 m/s
    cmd_vel_publisher_->publish(cmd_msg);
  }

  void stop_robot()
  {
    auto cmd_msg = geometry_msgs::msg::Twist();
    cmd_msg.linear.x = 0.0;
    cmd_vel_publisher_->publish(cmd_msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_subscriber_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MyRobotNode>());
  rclcpp::shutdown();
  return 0;
}
```

### Python Node Example

```python
#!/usr/bin/env python3
# src/my_robot_node_py.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class MyRobotNodePy(Node):
    def __init__(self):
        super().__init__('my_robot_node_py')

        # Create publisher
        self.cmd_vel_publisher = self.create_publisher(
            Twist, 'cmd_vel', 10)

        # Create subscriber
        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        # Create timer
        self.timer = self.create_timer(0.05, self.control_loop)  # 50ms

        self.get_logger().info('My Robot Node Py initialized')

    def laser_callback(self, msg):
        if len(msg.ranges) > 0:
            min_distance = min(msg.ranges)
            if min_distance < 1.0:
                self.stop_robot()

    def control_loop(self):
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.5  # Move forward at 0.5 m/s
        self.cmd_vel_publisher.publish(cmd_msg)

    def stop_robot(self):
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        self.cmd_vel_publisher.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MyRobotNodePy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Package Features

### Parameters in Packages

```cpp
// Parameter handling in C++
class ParameterNode : public rclcpp::Node
{
public:
  ParameterNode() : Node("parameter_node")
  {
    // Declare parameters with defaults
    this->declare_parameter("robot_name", "default_robot");
    this->declare_parameter("max_velocity", 1.0);
    this->declare_parameter("safety_distance", 0.5);

    // Get parameter values
    robot_name_ = this->get_parameter("robot_name").as_string();
    max_velocity_ = this->get_parameter("max_velocity").as_double();
    safety_distance_ = this->get_parameter("safety_distance").as_double();

    // Parameter callback
    this->set_on_parameters_set_callback(
      [this](const std::vector<rclcpp::Parameter> & parameters) {
        for (const auto & param : parameters) {
          if (param.get_name() == "max_velocity") {
            max_velocity_ = param.as_double();
          }
        }
        return rclcpp_interfaces::msg::SetParametersResult();
      });
  }

private:
  std::string robot_name_;
  double max_velocity_;
  double safety_distance_;
};
```

### Services in Packages

```cpp
// Service server implementation
#include "example_interfaces/srv/set_bool.hpp"

class RobotServiceNode : public rclcpp::Node
{
public:
  RobotServiceNode() : Node("robot_service_node")
  {
    service_ = this->create_service<example_interfaces::srv::SetBool>(
      "robot_enable",
      [this](const std::shared_ptr<example_interfaces::srv::SetBool::Request> request,
             std::shared_ptr<example_interfaces::srv::SetBool::Response> response) {
        enabled_ = request->data;
        response->success = true;
        response->message = enabled_ ? "Robot enabled" : "Robot disabled";
        RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
      });
  }

private:
  rclcpp::Service<example_interfaces::srv::SetBool>::SharedPtr service_;
  bool enabled_ = false;
};
```

## Package Management

### Workspace Organization

```
ros2_ws/                    # Root workspace
├── src/                    # Source directory
│   ├── my_robot_pkg/       # Your package
│   ├── external_pkg/       # External package
│   └── simulation_pkg/     # Simulation package
├── build/                  # Build artifacts
├── install/                # Installed packages
└── log/                    # Build logs
```

### Building Packages

```bash
# Build specific package
colcon build --packages-select my_robot_package

# Build with specific options
colcon build --packages-select my_robot_package --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build all packages
colcon build

# Source the workspace
source install/setup.bash
```

## Testing and Quality Assurance

### Unit Testing

```cpp
// test/test_my_robot.cpp
#include <gtest/gtest.h>
#include "my_robot_package/my_robot_class.hpp"

TEST(MyRobotTest, ConstructorTest) {
  MyRobotClass robot;
  EXPECT_TRUE(robot.is_initialized());
}

TEST(MyRobotTest, MovementTest) {
  MyRobotClass robot;
  EXPECT_EQ(robot.move_forward(1.0), true);
}
```

### Linting and Code Quality

```xml
<!-- package.xml additions for testing -->
<test_depend>ament_copyright</test_depend>
<test_depend>ament_flake8</test_depend>
<test_depend>ament_pep257</test_depend>
<test_depend>python3-pytest</test_depend>
```

## Package Distribution

### Creating Debian Packages

```bash
# Build Debian package
sudo apt install python3-rosdep python3-ament-package
sudo rosdep install --from-paths src --ignore-src -r -y
colcon build
sudo apt install ./install/my_robot_package/lib/my_robot_package/my_robot_package.deb
```

### Docker Integration

```dockerfile
# Dockerfile for ROS 2 package
FROM ros:humble

# Install dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-navigation2 \
    ros-humble-moveit \
    && rm -rf /var/lib/apt/lists/*

# Copy package
COPY . /ws/src/my_robot_package

# Build
RUN cd /ws && colcon build --packages-select my_robot_package

# Source and run
RUN echo "source /ws/install/setup.bash" >> ~/.bashrc
CMD ["bash", "-c", "source /ws/install/setup.bash && ros2 run my_robot_package my_robot_node"]
```

## Best Practices

### Package Structure
- Organize code by functionality (controllers, sensors, etc.)
- Use descriptive names following ROS naming conventions
- Include comprehensive documentation
- Separate concerns into different packages when appropriate

### Code Quality
- Follow ROS style guides (ament_uncrustify, ament_copyright)
- Write unit tests for all critical functionality
- Use parameter servers for configuration
- Implement proper error handling and logging

### Performance Considerations
- Minimize message copying
- Use shared pointers appropriately
- Consider real-time constraints
- Profile and optimize critical paths

## Integration with Physical AI Modules

ROS 2 packages serve as the building blocks for all Physical AI modules:
- **Simulation**: Gazebo integration packages
- **Perception**: Isaac ROS wrapper packages
- **Control**: Navigation and manipulation packages
- **VLA**: Natural language processing packages

## References

1. Open Robotics. (2023). *ROS 2 Package Development Guide*. https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html
2. Open Robotics. (2023). *ROS 2 Quality Assurance Tools*. https://docs.ros.org/en/humble/How-To-Guides/Ament-Copyright-And-Flake8.html
3. Quigley, M., et al. (2009). ROS: an open-source Robot Operating System. ICRA Workshop on Open Source Software.
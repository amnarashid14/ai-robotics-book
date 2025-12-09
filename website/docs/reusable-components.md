---
sidebar_position: 101
---

# Reusable Components for Code Examples and Technical Diagrams

## Overview

This document provides reusable components and templates for creating consistent code examples and technical diagrams throughout the Physical AI & Humanoid Robotics book. Standardized components ensure consistency and improve the learning experience.

## Code Example Templates

### C++ ROS 2 Node Template

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
// Include other message types as needed

class MyNode : public rclcpp::Node
{
public:
  MyNode() : Node("my_node_name")
  {
    // Create publisher
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic_name", 10);

    // Create subscriber
    subscription_ = this->create_subscription<std_msgs::msg::String>(
      "topic_name", 10,
      std::bind(&MyNode::topic_callback, this, std::placeholders::_1));

    // Create timer
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MyNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "Node initialized");
  }

private:
  void topic_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
  }

  void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "Hello World";
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MyNode>());
  rclcpp::shutdown();
  return 0;
}
```

### Python ROS 2 Node Template

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
# Import other message types as needed

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')

        # Create publisher
        self.publisher_ = self.create_publisher(String, 'topic_name', 10)

        # Create subscriber
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.topic_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Create timer
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('Node initialized')

    def topic_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.publisher_.publish(msg)

def main(args=None):
    rclcpp.init(args=args)
    my_node = MyNode()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Launch File Template

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='my_node',
            name='my_node_name',
            parameters=[
                # Parameter files or inline parameters
                {'param_name': 'param_value'},
                os.path.join(get_package_share_directory('my_robot_package'), 'config', 'params.yaml')
            ],
            remappings=[
                # Topic remappings if needed
                ('original_topic', 'remapped_topic')
            ],
            output='screen'  # Log output to screen
        ),
        # Add additional nodes as needed
    ])
```

## Technical Diagram Templates

### ASCII Architecture Diagrams

#### Basic ROS 2 Communication
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

#### Robot Software Stack
```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Navigation  │  │ Manipulation│  │ Perception          │  │
│  │ (MoveBase)  │  │ (MoveIt)    │  │ (OpenVINO, etc.)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    ROS 2 Middleware Layer                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Topics      │  │ Services    │  │ Actions             │  │
│  │ (Pub/Sub)   │  │ (Sync)      │  │ (Goal-Based)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Hardware Abstraction Layer                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Drivers     │  │ Sensors     │  │ Actuators           │  │
│  │ (Hardware   │  │ (Camera,    │  │ (Motors, Servos)    │  │
│  │ Interface)  │  │ LIDAR, etc.)│  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### System Integration Diagrams

#### Simulation to Real Deployment
```
Simulation Environment ──► Domain Randomization ──► Real Robot
         │                       │                      │
         │ (Isaac Sim, Gazebo)   │ (Variation)          │ (Physical)
         ▼                       ▼                      ▼
   Training Data        Robust Model        Deployment
   Generation           Learning             Validation
```

## Configuration Templates

### Package.xml Template

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Package description</description>
  <maintainer email="maintainer@example.com">Maintainer Name</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <!-- Add other dependencies as needed -->

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### CMakeLists.txt Template

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
# Add other dependencies as needed

# Add executables
add_executable(my_node src/my_node.cpp)
ament_target_dependencies(my_node rclcpp std_msgs)

# Install targets
install(TARGETS
  my_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install other files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## Documentation Templates

### API Documentation Template

```cpp
/**
 * @brief Brief description of the function/class
 *
 * @param[in] param1 Description of input parameter 1
 * @param[out] param2 Description of output parameter 2
 * @return Return value description
 *
 * @details Detailed description of the function/class
 *
 * @note Important notes about usage
 * @warning Potential issues or limitations
 *
 * @see Related functions or classes
 * @code
 * // Example usage
 * MyClass obj;
 * obj.my_function(input);
 * @endcode
 */
```

### Python Docstring Template

```python
def my_function(param1, param2=None):
    """
    Brief description of the function.

    Args:
        param1 (type): Description of param1
        param2 (type, optional): Description of param2. Defaults to None.

    Returns:
        type: Description of return value

    Raises:
        ExceptionType: Description of when this exception is raised

    Example:
        >>> my_function("input")
        output

    Note:
        Additional notes about the function
    """
    # Function implementation
    pass
```

## Common ROS 2 Patterns

### Parameter Declaration and Usage

```cpp
// C++ Parameter Usage
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

    // Parameter callback for dynamic reconfiguration
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

### TF2 Transform Usage

```cpp
// TF2 Transform Broadcaster
#include "tf2_ros/transform_broadcaster.h"

class TransformBroadcasterNode : public rclcpp::Node
{
public:
  TransformBroadcasterNode() : Node("transform_broadcaster_node")
  {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    timer_ = this->create_wall_timer(
      50ms, std::bind(&TransformBroadcasterNode::broadcast_transform, this));
  }

private:
  void broadcast_transform()
  {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->get_clock()->now();
    t.header.frame_id = "base_link";
    t.child_frame_id = "laser_frame";

    // Set transform values
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

## Testing Templates

### Unit Test Template

```cpp
#include <gtest/gtest.h>
#include "my_robot_package/my_class.hpp"

class MyClassTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Set up test fixtures
  }

  void TearDown() override
  {
    // Clean up after tests
  }

  MyClass test_object_;
};

TEST_F(MyClassTest, ConstructorTest)
{
  // Test constructor behavior
  EXPECT_TRUE(test_object_.is_initialized());
}

TEST_F(MyClassTest, MethodTest)
{
  // Test specific method
  auto result = test_object_.my_method();
  EXPECT_EQ(result, expected_value);
}
```

### Launch Test Template

```python
import launch
import launch_ros
import launch_testing
import pytest

@pytest.mark.launch_test
def generate_test_description():
    """Generate launch description with system under test."""
    node = launch_ros.actions.Node(
        package='my_robot_package',
        executable='my_node',
        name='my_node',
    )

    return launch.LaunchDescription([
        node,
        launch_testing.actions.ReadyToTest()
    ])

def test_node_launches(launch_service, proc_info, proc_output):
    """Test that the node launches without error."""
    launch_service("my_node")
    proc_info.assertWaitForShutdown(process="my_node", timeout=5)
```

## Quality Assurance Components

### Linting Configuration

```yaml
# .clang-format for C++ code
---
BasedOnStyle: Google
IndentWidth: 2
ColumnLimit: 100
AccessModifierOffset: -2
AlignAfterOpenBracket: Align
AlignConsecutiveAssignments: false
AlignConsecutiveDeclarations: false
AlignOperands: true
AlignTrailingComments: true
AllowAllParametersOfDeclarationOnNextLine: true
AllowShortBlocksOnASingleLine: false
AllowShortCaseLabelsOnASingleLine: false
AllowShortFunctionsOnASingleLine: All
AllowShortIfStatementsOnASingleLine: true
AllowShortLoopsOnASingleLine: true
...
```

### CI/CD Pipeline Template

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup ROS
      uses: ros-tooling/setup-ros@v0.7
      with:
        required-ros-distributions: humble
    - name: Test
      uses: ros-tooling/action-ros-ci@v0.3
      with:
        package-name: my_robot_package
        target-ros2-distro: humble
```

These reusable components provide a foundation for consistent documentation, code examples, and technical diagrams throughout the Physical AI & Humanoid Robotics book. Using these templates ensures consistency and reduces the time needed to create new content while maintaining high quality standards.
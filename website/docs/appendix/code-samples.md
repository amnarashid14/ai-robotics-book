---
sidebar_position: 2
---

# Code Samples & Troubleshooting

This appendix provides code samples and troubleshooting guidance for the implementations covered in the Physical AI & Humanoid Robotics book.

## ROS 2 Code Samples

### Basic Publisher Node (Python)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Basic Subscriber Node (Python)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS Perception Sample

### Image Processing Pipeline

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class IsaacImageProcessor(Node):

    def __init__(self):
        super().__init__('isaac_image_processor')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_processed',
            10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Apply Isaac-specific processing
        processed_image = self.apply_isaac_processing(cv_image)

        # Convert back to ROS Image message
        processed_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
        self.publisher.publish(processed_msg)

    def apply_isaac_processing(self, image):
        # Placeholder for Isaac-specific processing
        # This could include depth estimation, object detection, etc.
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def main(args=None):
    rclpy.init(args=args)
    processor = IsaacImageProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()
```

## Gazebo Integration Sample

### URDF Robot Model

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.083" ixy="0" ixz="0" iyy="0.083" iyz="0" izz="0.166"/>
    </inertial>
  </link>

  <!-- Camera Link -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Vision-Language-Action Integration

### Basic VLA Command Processing

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class VLARobotController(Node):

    def __init__(self):
        super().__init__('vla_robot_controller')
        self.subscription = self.create_subscription(
            String,
            'vla_command',
            self.command_callback,
            10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def command_callback(self, msg):
        command = msg.data.lower()

        twist = Twist()

        if 'forward' in command:
            twist.linear.x = 0.5
        elif 'backward' in command:
            twist.linear.x = -0.5
        elif 'left' in command:
            twist.angular.z = 0.5
        elif 'right' in command:
            twist.angular.z = -0.5
        elif 'stop' in command:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    controller = VLARobotController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
```

## Common Troubleshooting

### ROS 2 Issues

**Problem**: Nodes cannot communicate across machines
**Solution**: Ensure ROS_DOMAIN_ID is the same on all machines and firewall allows DDS traffic

**Problem**: High latency in real-time applications
**Solution**: Use appropriate QoS settings and consider real-time kernel

**Problem**: TF transform lookup fails
**Solution**: Check transform timestamps and ensure transform broadcaster is running

### Isaac ROS Issues

**Problem**: GPU memory errors
**Solution**: Monitor GPU memory usage and optimize batch sizes

**Problem**: Perception pipeline performance
**Solution**: Profile the pipeline and optimize bottlenecks

### Gazebo Issues

**Problem**: Simulation runs slower than real-time
**Solution**: Reduce physics update rate or simplify collision models

**Problem**: Robot falls through the ground
**Solution**: Check mass/inertia parameters and collision geometry

## Development Environment Setup

### Creating a ROS 2 Workspace

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build workspace
colcon build

# Source the workspace
source install/setup.bash
```

### Running Multiple Nodes

```bash
# Launch file example
# launch/my_robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_pkg',
            executable='publisher_node',
            name='publisher'
        ),
        Node(
            package='my_robot_pkg',
            executable='subscriber_node',
            name='subscriber'
        )
    ])
```

## Testing and Validation

### Unit Testing Example

```python
import unittest
import rclpy
from my_robot_pkg.robot_controller import RobotController

class TestRobotController(unittest.TestCase):

    def setUp(self):
        rclpy.init()
        self.controller = RobotController()

    def tearDown(self):
        self.controller.destroy_node()
        rclpy.shutdown()

    def test_move_forward(self):
        # Test forward movement
        self.controller.move_forward()
        # Add assertions here
```

## References

1. Open Robotics. (2023). *ROS 2 Tutorials*. https://docs.ros.org/en/humble/Tutorials.html
2. NVIDIA. (2023). *Isaac ROS Documentation*. https://nvidia-isaac-ros.github.io/
3. Open Source Robotics Foundation. (2023). *Gazebo Tutorials*. https://gazebosim.org/tutorials
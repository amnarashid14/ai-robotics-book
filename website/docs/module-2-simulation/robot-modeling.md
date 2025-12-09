---
sidebar_position: 3
---

# Robot Modeling & URDF/SDF

## Overview

This chapter covers robot modeling using Unified Robot Description Format (URDF) and Simulation Description Format (SDF), which are essential for describing robot structure, kinematics, and dynamics in simulation environments. Proper robot modeling is crucial for accurate simulation and successful sim-to-real transfer.

## Learning Objectives

By the end of this chapter, you will be able to:
- Create complete robot models using URDF for ROS 2 integration
- Develop SDF models for standalone Gazebo simulation
- Implement kinematic and dynamic properties for accurate simulation
- Validate robot models using simulation tools
- Optimize models for computational efficiency

## URDF Fundamentals

### URDF Structure

```
Robot Model (URDF)
├── Links (Rigid bodies)
│   ├── Visual: How the link looks
│   ├── Collision: How the link interacts physically
│   └── Inertial: Mass and inertia properties
├── Joints (Connections between links)
│   ├── Joint type: revolute, prismatic, fixed, etc.
│   ├── Limits: Position, velocity, effort limits
│   └── Origins: Position and orientation relative to parent
└── Transmissions: Motor and sensor interfaces (for control)
```

### Basic URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link (root of robot) -->
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
      <mass value="1.0"/>
      <inertia ixx="0.083" ixy="0" ixz="0" iyy="0.083" iyz="0" izz="0.166"/>
    </inertial>
  </link>

  <!-- Wheel links -->
  <link name="wheel_fl_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Joint connecting wheel to base -->
  <joint name="wheel_fl_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_fl_link"/>
    <origin xyz="0.2 0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
```

## Advanced URDF Features

### Xacro for Complex Models

Xacro (XML Macros) simplifies complex robot models:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="complex_robot">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.5" />
  <xacro:property name="base_height" value="0.2" />

  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix x_reflect y_reflect">
    <link name="${prefix}_wheel_link">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel_link"/>
      <origin xyz="${x_reflect*(base_length/2 - wheel_width/2)} ${y_reflect*(base_width/2)} 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.416" ixy="0" ixz="0" iyy="0.416" iyz="0" izz="0.833"/>
    </inertial>
  </link>

  <!-- Create wheels using macro -->
  <xacro:wheel prefix="front_left" x_reflect="1" y_reflect="1"/>
  <xacro:wheel prefix="front_right" x_reflect="1" y_reflect="-1"/>
  <xacro:wheel prefix="rear_left" x_reflect="-1" y_reflect="1"/>
  <xacro:wheel prefix="rear_right" x_reflect="-1" y_reflect="-1"/>
</robot>
```

### Inertial Properties

Proper inertial properties are crucial for accurate physics simulation:

```xml
<!-- Calculate inertial properties based on geometry -->
<link name="cylinder_link">
  <inertial>
    <!-- For a solid cylinder: Ixx = Iyy = m*(3*r² + h²)/12, Izz = m*r²/2 -->
    <mass value="1.0"/>
    <inertia
      ixx="0.009166"  <!-- 1.0*(3*0.1² + 0.2²)/12 -->
      ixy="0"
      ixz="0"
      iyy="0.009166"  <!-- same as ixx -->
      iyz="0"
      izz="0.005"/>   <!-- 1.0*0.1²/2 -->
  </inertial>
</link>
```

## SDF Format

### SDF vs URDF Comparison

While URDF is primarily used with ROS, SDF is Gazebo's native format:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="sdf_robot">
    <link name="base_link">
      <pose>0 0 0.1 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1667</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1667</iyy>
          <iyz>0</iyz>
          <izz>0.1667</izz>
        </inertia>
      </inertial>

      <visual name="base_visual">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>

      <collision name="base_collision">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
      </collision>
    </link>

    <!-- Joint definition in SDF -->
    <joint name="base_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>wheel_link</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>10</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
```

## Robot Model Validation

### URDF Validation Tools

```bash
# Check URDF for parsing errors
check_urdf /path/to/robot.urdf

# View robot model in RViz
ros2 run rviz2 rviz2

# Check joint limits and kinematics
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(cat robot.urdf)'
```

### Model Checking Node

```cpp
#include "rclcpp/rclcpp.hpp"
#include "urdf/model.h"

class URDFValidator : public rclcpp::Node
{
public:
  URDFValidator() : Node("urdf_validator")
  {
    std::string robot_description;
    this->declare_parameter("robot_description", "");
    this->get_parameter("robot_description", robot_description);

    if (!robot_description.empty()) {
      urdf::Model model;
      if (model.initString(robot_description)) {
        RCLCPP_INFO(this->get_logger(), "URDF parsed successfully");
        validate_model(model);
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF");
      }
    }
  }

private:
  void validate_model(const urdf::Model& model)
  {
    // Check for common issues
    if (model.getRoot() == nullptr) {
      RCLCPP_ERROR(this->get_logger(), "No root link found");
      return;
    }

    // Validate joint limits
    for (const auto& joint : model.joints_) {
      if (joint.second->type == urdf::Joint::REVOLUTE ||
          joint.second->type == urdf::Joint::PRISMATIC) {
        if (joint.second->limits == nullptr) {
          RCLCPP_WARN(this->get_logger(),
            "Joint %s has no limits defined", joint.first.c_str());
        }
      }
    }

    RCLCPP_INFO(this->get_logger(), "Model validation completed");
  }
};
```

## Kinematic Modeling

### Forward Kinematics

```cpp
#include "kdl/chain.hpp"
#include "kdl/chainfksolverpos_recursive.hpp"
#include "urdf/model.h"
#include "kdl_parser/kdl_parser.hpp"

class ForwardKinematicsSolver
{
public:
  ForwardKinematicsSolver(const std::string& robot_description)
  {
    urdf::Model model;
    model.initString(robot_description);

    KDL::Tree tree;
    if (kdl_parser::treeFromUrdfModel(model, tree)) {
      // Get chain from root to tip
      tree.getChain("base_link", "end_effector_link", chain_);
      fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
    }
  }

  bool calculateFK(const KDL::JntArray& joint_positions, KDL::Frame& end_effector_pose)
  {
    int result = fk_solver_->JntToCart(joint_positions, end_effector_pose);
    return (result >= 0);
  }

private:
  KDL::Chain chain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
};
```

### Inverse Kinematics

```cpp
#include "kdl/chainiksolverpos_nr.hpp"
#include "kdl/chainiksolvervel_pinv.hpp"

class InverseKinematicsSolver
{
public:
  InverseKinematicsSolver(const std::string& robot_description)
  {
    urdf::Model model;
    model.initString(robot_description);

    KDL::Tree tree;
    if (kdl_parser::treeFromUrdfModel(model, tree)) {
      tree.getChain("base_link", "end_effector_link", chain_);

      fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
      ik_vel_solver_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(chain_);
      ik_pos_solver_ = std::make_unique<KDL::ChainIkSolverPos_NR>(chain_,
        *fk_solver_, *ik_vel_solver_, 100, 1e-6);
    }
  }

  bool calculateIK(const KDL::Frame& end_effector_pose,
                   const KDL::JntArray& initial_positions,
                   KDL::JntArray& joint_positions)
  {
    int result = ik_pos_solver_->CartToJnt(initial_positions, end_effector_pose, joint_positions);
    return (result >= 0);
  }

private:
  KDL::Chain chain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
  std::unique_ptr<KDL::ChainIkSolverVel_pinv> ik_vel_solver_;
  std::unique_ptr<KDL::ChainIkSolverPos_NR> ik_pos_solver_;
};
```

## Dynamic Modeling

### Joint Dynamics

```xml
<!-- Joint with dynamics properties -->
<joint name="arm_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="forearm"/>
  <origin xyz="0 0 0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-2.0" upper="2.0" effort="100" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.01"/>
</joint>
```

### Transmission Interfaces

```xml
<!-- Transmission for ROS 2 control -->
<transmission name="wheel_trans_fl">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_fl_joint">
    <hardwareInterface>velocity_interface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor_fl">
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Gazebo Integration

### Gazebo-Specific Extensions

```xml
<!-- Gazebo-specific elements in URDF -->
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>

<!-- Gazebo plugins -->
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>robot</namespace>
      <remapping>cmd_vel:=cmd_vel</remapping>
      <remapping>odom:=odom</remapping>
    </ros>
    <update_rate>30</update_rate>
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.4</wheel_separation>
    <wheel_diameter>0.2</wheel_diameter>
  </plugin>
</gazebo>
```

## Model Optimization

### Simplification Strategies

```xml
<!-- Simplified collision geometry for performance -->
<link name="complex_visual_link">
  <!-- Detailed visual geometry -->
  <visual>
    <geometry>
      <mesh filename="package://my_robot/meshes/complex_shape.stl"/>
    </geometry>
  </visual>

  <!-- Simplified collision geometry -->
  <collision>
    <geometry>
      <box size="0.3 0.3 0.1"/>  <!-- Simple bounding box -->
    </geometry>
  </collision>

  <!-- Or use multiple simple shapes -->
  <collision>
    <geometry>
      <cylinder radius="0.1" length="0.2"/>
    </geometry>
  </collision>
</link>
```

### Level of Detail (LOD)

```xml
<!-- LOD for different simulation scenarios -->
<gazebo reference="sensor_mount">
  <!-- High detail for close-up simulation -->
  <static>true</static>
  <visual>
    <geometry>
      <mesh filename="package://my_robot/meshes/sensor_mount_high.stl"/>
    </geometry>
  </visual>

  <!-- Simplified version for performance -->
  <collision>
    <geometry>
      <box size="0.05 0.05 0.02"/>
    </geometry>
  </collision>
</gazebo>
```

## Multi-Robot Systems

### Robot Namespacing

```xml
<!-- Use Xacro properties for robot namespacing -->
<xacro:macro name="robot_model" params="name prefix:=robot">
  <xacro:include filename="$(find my_robot_description)/urdf/robot_base.urdf.xacro"/>

  <link name="${prefix}_base_link">
    <!-- Robot-specific elements -->
  </link>

  <!-- TF prefixing -->
  <gazebo>
    <plugin name="ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>${prefix}</robotNamespace>
    </plugin>
  </gazebo>
</xacro:macro>

<!-- Instantiate multiple robots -->
<xacro:robot_model name="robot1" prefix="robot1"/>
<xacro:robot_model name="robot2" prefix="robot2"/>
```

## Validation and Testing

### Model Validation Checklist

```cpp
class RobotModelValidator
{
public:
  struct ValidationResult {
    bool urdf_valid;
    bool kinematics_valid;
    bool dynamics_valid;
    bool collision_free;
    std::vector<std::string> issues;
  };

  ValidationResult validate_model(const std::string& urdf_string)
  {
    ValidationResult result;

    // 1. URDF parsing validation
    result.urdf_valid = validate_urdf_parsing(urdf_string);

    // 2. Kinematic chain validation
    result.kinematics_valid = validate_kinematic_chain(urdf_string);

    // 3. Dynamic properties validation
    result.dynamics_valid = validate_dynamic_properties(urdf_string);

    // 4. Collision detection validation
    result.collision_free = validate_self_collision(urdf_string);

    return result;
  }

private:
  bool validate_urdf_parsing(const std::string& urdf_string)
  {
    urdf::Model model;
    return model.initString(urdf_string);
  }

  bool validate_kinematic_chain(const std::string& urdf_string)
  {
    // Check for proper tree structure
    // Check for proper joint connections
    // Validate joint limits
    return true; // Implementation would go here
  }

  bool validate_dynamic_properties(const std::string& urdf_string)
  {
    // Check mass properties
    // Check inertia tensors
    // Validate joint dynamics
    return true; // Implementation would go here
  }

  bool validate_self_collision(const std::string& urdf_string)
  {
    // Check for self-collision at various configurations
    return true; // Implementation would go here
  }
};
```

## Troubleshooting Common Issues

### URDF Issues

1. **No Root Link**
   - **Symptom**: "No root link found" error
   - **Solution**: Ensure one link has no parent joint
   - **Prevention**: Start with a base_link and build outward

2. **Inertia Issues**
   - **Symptom**: Robot behaves unrealistically in simulation
   - **Solution**: Calculate proper inertia tensors
   - **Prevention**: Use CAD software to calculate mass properties

3. **Joint Limit Issues**
   - **Symptom**: Robot moves beyond physical limits
   - **Solution**: Define proper joint limits in URDF
   - **Prevention**: Check real robot specifications

### Gazebo Integration Issues

1. **Model Not Spawning**
   - **Symptom**: Robot doesn't appear in Gazebo
   - **Solution**: Check model path and SDF format
   - **Prevention**: Validate SDF with gz sdf -k

2. **Physics Issues**
   - **Symptom**: Robot falls through ground or explodes
   - **Solution**: Check mass/inertia and contact parameters
   - **Prevention**: Start with simple shapes and add complexity

## Best Practices

### Model Development
- Start with simple geometric shapes and add detail gradually
- Use proper mass and inertia properties from CAD models
- Include realistic joint limits and dynamics
- Test models in RViz before simulation

### Performance Optimization
- Use simplified collision geometry where possible
- Optimize mesh resolution for visual elements
- Use appropriate update rates for different components
- Implement level-of-detail for complex models

### Documentation
- Include model specifications and assumptions
- Document joint ranges and limitations
- Provide example configurations and use cases
- Maintain version control for model changes

## Integration with Isaac Sim

While URDF/SDF models work with Gazebo, Isaac Sim uses different formats:

- **URDF**: Convert to Isaac-compatible format using Omniverse
- **SDF**: May require conversion to USD format
- **Meshes**: Ensure compatibility with Isaac Sim's rendering pipeline

## References

1. Open Robotics. (2023). *URDF Documentation*. https://wiki.ros.org/urdf
2. Open Source Robotics Foundation. (2023). *SDF Specification*. https://sdformat.org/
3. Khatib, O., et al. (2018). *Robotics: Systems and Science*. MIT Press.
4. Siciliano, B., & Khatib, O. (2016). *Springer Handbook of Robotics*. Springer.
# Data Model: Physical AI & Humanoid Robotics Book

## Entities

### Module
- **Description**: A structured learning unit covering specific aspects of physical AI and robotics
- **Attributes**:
  - id: string (unique identifier)
  - title: string (e.g., "The Robotic Nervous System (ROS 2)")
  - description: string (detailed explanation of the module content)
  - learningObjectives: array of strings (specific skills and knowledge to be gained)
  - duration: integer (number of weeks)
  - toolsRequired: array of strings (software/hardware needed)
  - deliverables: array of strings (labs and assignments)

### CapstoneProject
- **Description**: The culminating project where students implement an autonomous humanoid robot system
- **Attributes**:
  - id: string (unique identifier)
  - title: string ("The Autonomous Humanoid")
  - description: string (detailed explanation of the capstone requirements)
  - requirements: array of strings (specific functional requirements)
  - successCriteria: array of strings (measurable outcomes)
  - evaluationRubric: object (grading criteria and weights)

### HardwareComponent
- **Description**: Physical and virtual components including workstations, Jetson kits, sensors, and actuators
- **Attributes**:
  - id: string (unique identifier)
  - name: string (e.g., "NVIDIA Jetson AGX Orin 64GB")
  - type: string (e.g., "edge-compute", "sensor", "actuator", "workstation")
  - specifications: object (technical specs like performance, memory, etc.)
  - purpose: string (what it's used for)
  - cost: string (approximate cost)

### SimulationEnvironment
- **Description**: Digital twin platforms (Gazebo, Unity) for testing and development
- **Attributes**:
  - id: string (unique identifier)
  - name: string ("Gazebo" or "Unity")
  - type: string ("simulation-platform")
  - capabilities: array of strings (what it can simulate)
  - integration: string (how it connects to ROS 2)
  - useCases: array of strings (when to use this platform)

### AIModel
- **Description**: Vision-Language-Action models that enable perception and action in physical environments
- **Attributes**:
  - id: string (unique identifier)
  - name: string (e.g., "Vision-Language-Action Model")
  - type: string (e.g., "VLA", "perception", "planning")
  - inputs: array of strings (data types it accepts)
  - outputs: array of strings (actions or decisions it produces)
  - requirements: array of strings (compute, memory, etc.)

## Relationships

- Module contains many Deliverable/Lab
- CapstoneProject uses many Module (integration of all modules)
- HardwareComponent supports Module (provides necessary tools)
- SimulationEnvironment integrates with Module (for simulation aspects)
- AIModel operates within Module (for VLA aspects)

## Validation Rules

- Module duration must be between 1-4 weeks
- Module must have at least one learning objective
- HardwareComponent cost must be provided as an estimate
- CapstoneProject must reference all four modules
- AIModel inputs/outputs must align with ROS 2 message types where applicable
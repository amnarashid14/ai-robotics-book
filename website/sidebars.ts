import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar configuration for the Physical AI & Humanoid Robotics book
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part I: Foundations',
      items: [
        'foundations/intro',
        'foundations/robotics-fundamentals',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part II: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros/intro',
        'module-1-ros/architecture',
        'module-1-ros/packages',
        'module-1-ros/tf-transforms',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part III: Digital Twin Simulation',
      items: [
        'module-2-simulation/intro',
        'module-2-simulation/gazebo',
        'module-2-simulation/robot-modeling',
        'module-2-simulation/unity',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part IV: AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-isaac/intro',
        'module-3-isaac/perception',
        'module-3-isaac/planning',
        'module-3-isaac/gpu-inference',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part V: Vision-Language-Action Systems',
      items: [
        'module-4-vla/intro',
        'module-4-vla/vla-models',
        'module-4-vla/multimodal',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part VI: Capstone Project',
      items: [
        'capstone/intro',
        'capstone/autonomous-humanoid',
        'capstone/sim-to-real',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendix/hardware',
        'appendix/code-samples',
        'appendix/references',
      ],
      collapsed: true,
    },
    {
      type: 'doc',
      id: 'conclusion',
      label: 'Conclusion',
    },
  ],
};

export default sidebars;

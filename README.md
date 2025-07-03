
# NeuroBot ML for RL Training

This repository contains a suite of machine learning and reinforcement learning projects designed for robotic control and signal processing, focusing on neuro-robotics applications and simulated environments.

---

## Project Overview

The main objective of this project is to explore **robotic control through machine learning**, including reinforcement learning (RL) for path planning, signal processing for brain-computer interfaces (BCI), and predictive modeling for robotic performance. The codebase demonstrates:

- Training and evaluation of RL agents for 3D path planning in dynamic environments.
- Signal denoising and feature extraction from simulated EEG data.
- Brain command decoding using neural networks on synthetic EEG signals.
- Predictive modeling of user accuracy and system latency impacts.
- Visualization of trajectories, classification boundaries, and model evaluation.

---

## Repository Structure 

### 1. 
- Custom OpenAI Gym environment simulating a 3D robot navigating a grid with discrete actions (move forward, turn left, right, stop).  
- Uses **Proximal Policy Optimization (PPO)** from Stable Baselines3 to train the agent.  
- Visualizes the trained robot path against an optimal path in 3D.

### 2. 
- Generates a synthetic 2D classification dataset.  
- Trains and compares three models: Support Vector Machine (SVM), Deep Neural Network (DNN), and a simple Convolutional Neural Network (CNN).  
- Visualizes decision boundaries for all three classifiers.

### 3.   
- Visualizes the relationship between **communication latency** and **robotic control accuracy**.  
- Illustrates how increased latency degrades control performance, important for remote robotic systems.

### 4.   
- Simulates and forecasts user accuracy over multiple sessions using **autoregression (AR)** time-series modeling.  
- Plots actual vs predicted user accuracy improvement.

### 5. 
- Compares **predicted vs actual robot trajectories** in 2D space with simulated noise.  
- Highlights prediction errors using a color map on the plotted trajectory.

### 6. `  
- Demonstrates **EEG signal denoising** using **Independent Component Analysis (ICA)**.  
- Calculates and compares Signal-to-Noise Ratio (SNR) before and after denoising for multiple EEG channels.  
- Visualizes SNR improvements and their distributions.

### 7.   
- Simulates EEG data for three brain commands: Move Forward, Turn Left, Stop.  
- Trains a **Multi-Layer Perceptron (MLP)** classifier to decode these commands.  
- Visualizes classification performance via a confusion matrix and plots predicted robot movements.

🤖 Requirements: the main Python packages used in this project:

numpy

matplotlib

seaborn

scikit-learn

tensorflow

stable-baselines3

statsmodels

gym

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mohima6/neurobot-ml-for-paperwork-rl-training.git
   cd neurobot-ml-for-paperwork-rl-training
   
2. Install required Python packages:
    ```bash
    pip install -r requirements.txt


Future Work: 

Extend RL environment for more complex multi-agent navigation.

Implement transfer learning for improved brain command decoding.

Explore advanced signal denoising techniques and feature extraction.


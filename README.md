# Simple LQR Controller

This repository contains a simple implementation of a Linear Quadratic Regulator (LQR) controller. The LQR controller is a state-feedback controller that minimizes a quadratic cost function of the state and control inputs.

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Control

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

The `LQRController` class provides a simple interface for designing and implementing LQR controllers:

```python
from lqr_controller import LQRController

# Define system matrices
A = ... # System dynamics matrix
B = ... # Input matrix
Q = ... # State cost matrix
R = ... # Input cost matrix

# Create LQR controller
controller = LQRController(A, B, Q, R)

# Compute optimal feedback gain
K = controller.compute_gain()

# Get control input for a given state
state = ... # Current state vector
u = controller.get_control_input(state)
```

## Features

- Continuous-time LQR design
- Optimal feedback gain computation
- State feedback control
- Stability checks
- Cost matrix validation

## Theory

The LQR controller minimizes the infinite horizon cost function:

\[J = \int_0^\infty (x^T Q x + u^T R u) dt\]

where:
- x is the state vector
- u is the control input vector
- Q is the state cost matrix (positive semi-definite)
- R is the input cost matrix (positive definite)

The resulting optimal control law is:
\[u = -Kx\]

where K is the optimal feedback gain matrix. 
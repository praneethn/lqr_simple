import numpy as np
from lqr_controller import LQRController

def main():
    """
    Example of using the LQR controller for a simple second-order system.
    The system represents a mass-spring-damper system with:
    - Position and velocity states
    - Force input
    """
    # System parameters
    m = 1.0  # mass (kg)
    k = 1.0  # spring constant (N/m)
    b = 0.1  # damping coefficient (N⋅s/m)

    # System matrices for state-space representation
    # States: [position, velocity]
    # Input: force
    A = np.array([
        [0, 1],
        [-k/m, -b/m]
    ])
    B = np.array([
        [0],
        [1/m]
    ])

    # Cost matrices
    Q = np.array([
        [1.0, 0.0],  # Penalize position error
        [0.0, 0.1]   # Penalize velocity error
    ])
    R = np.array([[0.1]])  # Penalize control effort

    # Create and initialize the LQR controller
    controller = LQRController(A, B, Q, R)

    # Compute the optimal feedback gain
    K = controller.compute_gain()
    print("Optimal feedback gain K:")
    print(K)

    # Check if the closed-loop system is stable
    if controller.is_stable():
        print("\nThe closed-loop system is stable!")
    else:
        print("\nWarning: The closed-loop system is unstable!")

    # Simulate the system response for an initial state
    initial_state = np.array([1.0, 0.0])  # Initial position = 1m, velocity = 0 m/s
    u = controller.get_control_input(initial_state)
    print("\nControl input for initial state:")
    print(u)

    # Get the closed-loop system matrix
    A_cl = controller.get_closed_loop_system()
    print("\nClosed-loop system matrix:")
    print(A_cl)

if __name__ == "__main__":
    main() 
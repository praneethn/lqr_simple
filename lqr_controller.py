import numpy as np
from scipy import linalg
import control

class LQRController:
    """
    A simple Linear Quadratic Regulator (LQR) controller implementation.
    
    This class implements a continuous-time LQR controller for linear time-invariant systems
    of the form: dx/dt = Ax + Bu
    """
    
    def __init__(self, A, B, Q, R):
        """
        Initialize the LQR controller.
        
        Args:
            A (numpy.ndarray): System dynamics matrix (n x n)
            B (numpy.ndarray): Input matrix (n x m)
            Q (numpy.ndarray): State cost matrix (n x n)
            R (numpy.ndarray): Input cost matrix (m x m)
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.K = None
        
        # Validate dimensions
        self._validate_dimensions()
        # Validate cost matrices
        self._validate_cost_matrices()
        
    def _validate_dimensions(self):
        """Validate the dimensions of the system matrices."""
        n = self.A.shape[0]  # State dimension
        
        if self.A.shape[1] != n:
            raise ValueError(f"A matrix must be square, got shape {self.A.shape}")
        
        if self.B.shape[0] != n:
            raise ValueError(f"B matrix must have {n} rows, got shape {self.B.shape}")
        
        m = self.B.shape[1]  # Input dimension
        
        if self.Q.shape != (n, n):
            raise ValueError(f"Q matrix must be {n}x{n}, got shape {self.Q.shape}")
            
        if self.R.shape != (m, m):
            raise ValueError(f"R matrix must be {m}x{m}, got shape {self.R.shape}")
    
    def _validate_cost_matrices(self):
        """Validate that Q is positive semi-definite and R is positive definite."""
        # Check if Q is positive semi-definite
        Q_eigvals = np.linalg.eigvals(self.Q)
        if not np.all(Q_eigvals >= -1e-10):  # Allow for numerical errors
            raise ValueError("Q matrix must be positive semi-definite")
        
        # Check if R is positive definite
        R_eigvals = np.linalg.eigvals(self.R)
        if not np.all(R_eigvals > 0):
            raise ValueError("R matrix must be positive definite")
    
    def is_stabilizable(self):
        """
        Check if the system is stabilizable using the PBH test.
        
        Returns:
            bool: True if the system is stabilizable, False otherwise
        """
        n = self.A.shape[0]
        
        # Check rank of [sI - A, B] for all unstable eigenvalues
        eigvals = np.linalg.eigvals(self.A)
        for eigval in eigvals:
            if eigval.real >= 0:  # Check only unstable eigenvalues
                # Form the PBH matrix
                pbh_matrix = np.hstack([
                    eigval * np.eye(n) - self.A,
                    self.B
                ])
                if np.linalg.matrix_rank(pbh_matrix) < n:
                    return False
        return True
    
    def compute_gain(self):
        """
        Compute the optimal feedback gain matrix K.
        
        Returns:
            numpy.ndarray: Optimal feedback gain matrix K
        """
        if not self.is_stabilizable():
            raise ValueError("System is not stabilizable")
        
        # Solve the continuous-time algebraic Riccati equation
        P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        
        # Compute the optimal gain matrix
        self.K = np.linalg.solve(self.R, self.B.T @ P)
        
        return self.K
    
    def get_control_input(self, state):
        """
        Compute the control input for a given state.
        
        Args:
            state (numpy.ndarray): Current state vector
            
        Returns:
            numpy.ndarray: Optimal control input
        """
        if self.K is None:
            self.compute_gain()
        
        return -self.K @ state
    
    def get_closed_loop_system(self):
        """
        Get the closed-loop system matrix A - BK.
        
        Returns:
            numpy.ndarray: Closed-loop system matrix
        """
        if self.K is None:
            self.compute_gain()
            
        return self.A - self.B @ self.K
    
    def is_stable(self):
        """
        Check if the closed-loop system is stable.
        
        Returns:
            bool: True if the closed-loop system is stable, False otherwise
        """
        closed_loop_matrix = self.get_closed_loop_system()
        eigenvalues = np.linalg.eigvals(closed_loop_matrix)
        return np.all(eigenvalues.real < 0) 
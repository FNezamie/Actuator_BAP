import numpy as np
import math

class EKF_BallTracker:
    def __init__(self,
                 initial_y, initial_vy,
                 initial_P_yy, initial_P_vv,
                 process_noise_y, process_noise_vy,
                 measurement_noise_y,
                 dt, g_pix, k_drag_pix, px_per_m):
        self.px_per_m = px_per_m

        # State vector [y, vy]^T
        self.x_hat = np.array([initial_y, initial_vy], dtype=float)

        # State covariance matrix P_k|k
        self.P = np.diag([initial_P_yy, initial_P_vv]).astype(float)

        # Process noise covariance Q
        self.Q = np.diag([process_noise_y, process_noise_vy]).astype(float)

        # Measurement noise covariance R 
        self.R = float(measurement_noise_y)

        # Time step
        self.dt = float(dt)

        # Physical parameters (in pixel units)
        self.g_pix = float(g_pix)
        self.k_drag_pix = float(k_drag_pix)  # Units: 1/pixel

        # Measurement matrix H (constant)
        self.H = np.array([[1, 0]], dtype=float)
        self.H_T = self.H.T

        # Identity matrix
        self._I = np.eye(2)

    def predict(self):
        y_prev, vy_prev = self.x_hat
        a_net = self.g_pix - self.k_drag_pix * vy_prev**2 # Simple quadratic drag model

        # State prediction
        y_pred = y_prev + vy_prev * self.dt + 0.5 * a_net * self.dt**2
        vy_pred = vy_prev + a_net * self.dt
        self.x_hat = np.array([y_pred, vy_pred])

        F = np.zeros((2, 2), dtype=float)
        F[0, 0] = 1.0
        F[0, 1] = self.dt - self.k_drag_pix * vy_prev * self.dt**2 # Use vy_prev from start of interval
        F[1, 0] = 0.0
        F[1, 1] = 1.0 - 2.0 * self.k_drag_pix * vy_prev * self.dt # Use vy_prev from start of interval

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q
        return self.x_hat

    def update(self, measurement_y):
        # Innovation (measurement residual)
        y_tilde = measurement_y - self.x_hat[0] # x_hat[0] is y_pred_k|k-1

        # Innovation covariance
        # S = H P_k|k-1 H^T + R
        # Since H = [1, 0], H P H^T = P[0,0]
        S = self.P[0, 0] + self.R
        if S == 0: # Avoid division by zero
            S = 1e-9 # Add a tiny epsilon

        # Kalman gain K
        # K = P_k|k-1 H^T S^-1
        # P_k|k-1 H^T = [P[0,0], P[1,0]]^T
        K_column_vector = self.P @ self.H_T
        K = K_column_vector / S

        # State update
        self.x_hat = self.x_hat + (K * y_tilde).flatten()

        # Covariance update 
        self.P = (self._I - K @ self.H) @ self.P
        # Ensure P remains symmetric
        self.P = 0.5 * (self.P + self.P.T)
        return self.x_hat

    def get_current_position(self):
        return self.x_hat[0]

    def get_current_velocity(self):
        return self.x_hat[1] / self.px_per_m

    def get_predicted(self):
        """
        Predicts next-step position based on the current updated state x_hat (k|k).
        This is y_hat (k+1|k).
        """
        A = 8
        y_curr_updated, vy_curr_updated = self.x_hat # These are x_k|k
        a_net_curr_updated = self.g_pix - self.k_drag_pix * vy_curr_updated**2
        return y_curr_updated + vy_curr_updated * A * self.dt + 0.5 * a_net_curr_updated * A * self.dt**2

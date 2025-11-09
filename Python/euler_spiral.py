#!/usr/bin/env python3
"""
Euler Spiral Clock Spring Implementation

This module implements a clock spring using Euler spirals (clothoids) which provide
natural arc length parameterization and smooth curvature transitions. Unlike 
Archimedean spirals, Euler spirals can satisfy endpoint and length constraints
simultaneously through numerical optimization.

The Euler spiral has curvature κ(s) = s/a² where s is arc length and a is 
the spiral parameter. This gives parametric equations using Fresnel integrals.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import fresnel
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional
import sympy as sp


class EulerSpiralCurve:
    """
    Euler spiral implementation for clock spring with exact arc length control.
    
    The Euler spiral (clothoid) has linearly varying curvature, making it ideal
    for smooth cable routing with precise length constraints.
    """
    
    def __init__(self):
        """Initialize the Euler spiral curve."""
        self.spiral_param = None  # 'a' parameter
        self.max_arc_length = None  # Maximum arc length
        self.start_point = None
        self.end_point = None
        self.rotation_offset = 0.0  # Overall rotation of the spiral
        
    def fresnel_integrals(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fresnel integrals C(t) and S(t).
        
        Args:
            t: Input parameter array
            
        Returns:
            Tuple of (C(t), S(t)) arrays
        """
        # scipy.special.fresnel returns (S, C), we want (C, S)
        S, C = fresnel(t)
        return C, S
    
    def euler_spiral_point(self, s: float, a: float, 
                          x0: float = 0, y0: float = 0, 
                          theta0: float = 0) -> Tuple[float, float, float]:
        """
        Compute a point on the Euler spiral at arc length s.
        
        Args:
            s: Arc length parameter
            a: Spiral parameter (controls tightness)
            x0, y0: Starting position
            theta0: Starting angle
            
        Returns:
            Tuple of (x, y, theta) where theta is the angle at that point
        """
        if s == 0:
            return x0, y0, theta0
            
        # Normalized parameter for Fresnel integrals
        t = s / (a * np.sqrt(np.pi/2))
        
        # Compute Fresnel integrals
        C, S = self.fresnel_integrals(t)
        
        # Scale by spiral parameter
        scale = a * np.sqrt(np.pi/2)
        
        # Compute position relative to start
        dx = scale * C
        dy = scale * S
        
        # Apply starting rotation and translation
        cos_theta0 = np.cos(theta0)
        sin_theta0 = np.sin(theta0)
        
        x = x0 + dx * cos_theta0 - dy * sin_theta0
        y = y0 + dx * sin_theta0 + dy * cos_theta0
        
        # Angle at this point
        theta = theta0 + s**2 / (2 * a**2)
        
        return x, y, theta
    
    def generate_spiral_points(self, s_max: float, a: float, 
                             x0: float = 0, y0: float = 0, 
                             theta0: float = 0, 
                             num_points: int = 200) -> np.ndarray:
        """
        Generate points along the Euler spiral.
        
        Args:
            s_max: Maximum arc length
            a: Spiral parameter
            x0, y0: Starting position
            theta0: Starting angle
            num_points: Number of points to generate
            
        Returns:
            Array of shape (num_points, 3) with [x, y, z] coordinates
        """
        s_values = np.linspace(0, s_max, num_points)
        points = np.zeros((num_points, 3))
        
        for i, s in enumerate(s_values):
            x, y, theta = self.euler_spiral_point(s, a, x0, y0, theta0)
            points[i] = [x, y, 0]  # z = 0 for planar curve
            
        return points
    
    def constraint_error(self, params: np.ndarray, 
                        start_point: Tuple[float, float],
                        end_point: Tuple[float, float],
                        target_length: float) -> float:
        """
        Compute error for constraint optimization.
        
        Args:
            params: [a, s_max, theta0] - spiral parameter, arc length, start angle
            start_point: (x0, y0) starting position
            end_point: (x_target, y_target) target ending position
            target_length: Desired arc length
            
        Returns:
            Total constraint violation error
        """
        a, s_max, theta0 = params
        x0, y0 = start_point
        x_target, y_target = end_point
        
        # Ensure positive parameters
        if a <= 0 or s_max <= 0:
            return 1e6
        
        # Compute end point of spiral
        x_end, y_end, _ = self.euler_spiral_point(s_max, a, x0, y0, theta0)
        
        # Endpoint error
        endpoint_error = (x_end - x_target)**2 + (y_end - y_target)**2
        
        # Length error (s_max should equal target_length)
        length_error = (s_max - target_length)**2
        
        # Total weighted error
        return endpoint_error + 0.1 * length_error
    
    def fit_to_constraints(self, start_point: Tuple[float, float],
                          end_point: Tuple[float, float],
                          target_length: float) -> Dict:
        """
        Fit Euler spiral to satisfy endpoint and length constraints.
        
        Args:
            start_point: (x0, y0) starting position
            end_point: (x_target, y_target) ending position  
            target_length: Desired arc length
            
        Returns:
            Dictionary with optimized parameters and success info
        """
        # Initial guess
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        straight_dist = np.sqrt(dx**2 + dy**2)
        
        # Initial parameter estimates
        a_init = target_length / (2 * np.pi)  # Rough estimate
        s_max_init = target_length
        theta0_init = np.arctan2(dy, dx)  # Point toward target
        
        initial_params = [a_init, s_max_init, theta0_init]
        
        # Bounds: a > 0, s_max > 0, theta0 unrestricted
        bounds = [(0.1, None), (straight_dist, target_length * 2), (-np.pi, np.pi)]
        
        # Optimize
        result = minimize(
            self.constraint_error,
            initial_params,
            args=(start_point, end_point, target_length),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return {
            'success': result.success,
            'a': result.x[0],
            's_max': result.x[1], 
            'theta0': result.x[2],
            'error': result.fun,
            'message': result.message
        }
    
    def create_clock_spring(self, r_inner: float, r_outer: float,
                           theta_rotor: float, target_length: float,
                           num_points: int = 200) -> Tuple[np.ndarray, Dict]:
        """
        Create a clock spring curve using Euler spiral.
        
        Args:
            r_inner: Inner radius (stator position)
            r_outer: Outer radius
            theta_rotor: Rotor angular position
            target_length: Desired cable length
            num_points: Number of points to generate
            
        Returns:
            Tuple of (points array, optimization info dict)
        """
        # Define constraint points
        start_point = (r_inner, 0.0)  # Stator always at (r_inner, 0)
        end_point = (r_outer * np.cos(theta_rotor), 
                    r_outer * np.sin(theta_rotor))  # Rotor position
        
        print(f"Fitting Euler spiral:")
        print(f"  Start: {start_point}")
        print(f"  End: {end_point}")
        print(f"  Target length: {target_length} mm")
        
        # Fit spiral to constraints
        opt_result = self.fit_to_constraints(start_point, end_point, target_length)
        
        if not opt_result['success']:
            print(f"Warning: Optimization failed: {opt_result['message']}")
        
        print(f"  Optimized parameters:")
        print(f"    a = {opt_result['a']:.3f}")
        print(f"    s_max = {opt_result['s_max']:.3f}")
        print(f"    theta0 = {opt_result['theta0']:.3f} rad")
        print(f"    Error = {opt_result['error']:.6f}")
        
        # Generate spiral points
        points = self.generate_spiral_points(
            opt_result['s_max'], opt_result['a'],
            start_point[0], start_point[1], opt_result['theta0'],
            num_points
        )
        
        return points, opt_result
    
    def plot_curve(self, points: np.ndarray, 
                   figsize: Tuple[float, float] = (10, 8),
                   show_plot: bool = True) -> plt.Figure:
        """
        Plot the Euler spiral curve.
        
        Args:
            points: Array of curve points
            figsize: Figure size
            show_plot: Whether to display the plot
            
        Returns:
            The matplotlib Figure object
        """
        x_points, y_points, z_points = points[:, 0], points[:, 1], points[:, 2]
        
        # Calculate actual length
        length = 0
        for i in range(1, len(points)):
            dx = x_points[i] - x_points[i-1] 
            dy = y_points[i] - y_points[i-1]
            dz = z_points[i] - z_points[i-1]
            length += np.sqrt(dx**2 + dy**2 + dz**2)
        
        print(f"Actual curve length: {length:.3f} mm")
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the curve
        ax.plot(x_points, y_points, z_points, 'b-', linewidth=2, label='Euler Spiral')
        
        # Mark start and end points
        ax.scatter([x_points[0]], [y_points[0]], [z_points[0]], 
                  color='green', s=100, label='Stator', marker='o')
        ax.scatter([x_points[-1]], [y_points[-1]], [z_points[-1]], 
                  color='red', s=100, label='Rotor', marker='s')
        
        # Set equal aspect ratio
        max_range = np.array([x_points.max()-x_points.min(), 
                             y_points.max()-y_points.min(),
                             z_points.max()-z_points.min()]).max() / 2.0
        
        mid_x = (x_points.max()+x_points.min()) * 0.5
        mid_y = (y_points.max()+y_points.min()) * 0.5
        mid_z = (z_points.max()+z_points.min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Clock Spring - Euler Spiral')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
            
        return fig


def main():
    """Example usage of the Euler spiral clock spring."""
    
    print("Euler Spiral Clock Spring Generator")
    print("=" * 40)
    
    # Create Euler spiral instance
    spiral = EulerSpiralCurve()
    
    # Clock spring parameters
    r_inner = 40.0      # Inner radius (mm)
    r_outer = 30.0      # Outer radius (mm)  
    theta_rotor = 0  # Rotor at 60 degrees
    target_length = 200.0  # Desired cable length (mm)
    
    print(f"Clock spring parameters:")
    print(f"  Inner radius: {r_inner} mm")
    print(f"  Outer radius: {r_outer} mm")
    print(f"  Rotor angle: {theta_rotor:.3f} rad ({np.degrees(theta_rotor):.1f}°)")
    print(f"  Target length: {target_length} mm")
    print()
    
    # Generate the curve
    points, opt_info = spiral.create_clock_spring(
        r_inner, r_outer, theta_rotor, target_length, num_points=300
    )
    
    print(f"\nGenerated {len(points)} points along the curve")
    print("First 3 points [x, y, z]:")
    for i in range(3):
        print(f"  {points[i]}")
    print("Last 3 points [x, y, z]:")
    for i in range(-3, 0):
        print(f"  {points[i]}")
    print()
    
    # Plot the curve
    print("Generating plot...")
    spiral.plot_curve(points)


if __name__ == "__main__":
    main()
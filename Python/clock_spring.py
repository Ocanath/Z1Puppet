#!/usr/bin/env python3
"""
Clock Spring Wire Harness Parametric Curve Generator

This module generates parametric equations for modeling the curve of a 'clock spring' 
wire harness using SymPy. The clock spring follows a planar Archimedean spiral
that transitions from an inner radius to an outer radius based on rotation angle.
Designed for flat cables such as flex PCB circuits or ribbon cables.

Parameters:
- r_inner: Inner radius of the clock spring
- r_outer: Outer radius of the clock spring  
- theta_max: Maximum rotation angle
- theta_offset: Rotor angular position (stator always fixed at theta=0)
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List


class ClockSpringCurve:
    """
    Parametric representation of a clock spring wire harness curve.
    
    The clock spring is modeled as a planar Archimedean spiral that transitions
    from inner radius to outer radius over a specified angular range.
    Designed for flat cables with strain relief at endpoints.
    Stator (inner) end is always at theta=0, rotor (outer) end position varies with theta_offset.
    """
    
    def __init__(self):
        # Define symbolic parameters
        self.t = sp.Symbol('t', real=True)  # Parameter variable (0 to 1)
        self.r_inner = sp.Symbol('r_inner', positive=True)  # Inner radius
        self.r_outer = sp.Symbol('r_outer', positive=True)  # Outer radius
        self.theta_max = sp.Symbol('theta_max', positive=True)  # Maximum rotation angle
        self.theta_offset = sp.Symbol('theta_offset', real=True)  # Rotor position (stator fixed at theta=0)
        
        
        # Generate parametric equations
        self._generate_equations()
    
    def _generate_equations(self):
        """Generate the parametric equations for the clock spring curve."""
        
        # Radius varies linearly from inner to outer
        self.radius = self.r_inner + (self.r_outer - self.r_inner) * self.t
        
        # Angle: stator at theta=0 (t=0), rotor at theta_offset + theta_max (t=1)
        # This keeps stator fixed while rotor position changes with theta_offset
        self.angle = self.theta_max * self.t + self.theta_offset * self.t
        
        # Cartesian coordinates (planar curve, z=0)
        self.x = self.radius * sp.cos(self.angle)
        self.y = self.radius * sp.sin(self.angle)
        self.z = sp.Integer(0)
        
        # Parametric curve vector
        self.curve = sp.Matrix([self.x, self.y, self.z])
    
    def get_equations(self) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
        """
        Get the parametric equations as SymPy expressions.
        
        Returns:
            Tuple of (x(t), y(t), z(t)) expressions
        """
        return self.x, self.y, self.z
    
    def get_curve_vector(self) -> sp.Matrix:
        """
        Get the parametric curve as a vector.
        
        Returns:
            SymPy Matrix representing the curve vector [x(t), y(t), z(t)]
        """
        return self.curve
    
    def substitute_values(self, values: dict) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
        """
        Substitute numerical values for the parameters.
        
        Args:
            values: Dictionary mapping parameter symbols to numerical values
            
        Returns:
            Tuple of parametric equations with substituted values
        """
        x_sub = self.x.subs(values)
        y_sub = self.y.subs(values)
        z_sub = self.z.subs(values)
        
        return x_sub, y_sub, z_sub
    
    def generate_points(self, values: dict, num_points: int = 100) -> np.ndarray:
        """
        Generate numerical points along the clock spring curve.
        
        Args:
            values: Dictionary of parameter values
            num_points: Number of points to generate along the curve
            
        Returns:
            NumPy array of shape (num_points, 3) containing [x, y, z] coordinates
        """
        # Substitute values into equations
        x_func, y_func, z_func = self.substitute_values(values)
        
        # Convert to numerical functions
        x_lambdified = sp.lambdify(self.t, x_func, 'numpy')
        y_lambdified = sp.lambdify(self.t, y_func, 'numpy')
        z_lambdified = sp.lambdify(self.t, z_func, 'numpy')
        
        # Generate parameter values
        t_values = np.linspace(0, 1, num_points)
        
        # Calculate coordinates
        x_points = x_lambdified(t_values)
        y_points = y_lambdified(t_values)
        z_points = np.zeros_like(x_points)  # Create array of zeros same size as x_points
        
        return np.column_stack([x_points, y_points, z_points])
    
    def plot_curve(self, values: dict, num_points: int = 200, 
                   figsize: Tuple[float, float] = (10, 8),
                   show_plot: bool = True) -> plt.Figure:
        """
        Plot the 3D clock spring curve with correct aspect ratio.
        
        Args:
            values: Dictionary of parameter values
            num_points: Number of points to generate for smooth curve
            figsize: Figure size as (width, height)
            show_plot: Whether to display the plot
            
        Returns:
            The matplotlib Figure object
        """
        # Generate points
        points = self.generate_points(values, num_points)
        x_points, y_points, z_points = points[:, 0], points[:, 1], points[:, 2]
        length = 0
        for i in range(1,num_points):
            prev_i = i-1
            dx = x_points[i] - x_points[prev_i]
            dy = y_points[i] - y_points[prev_i]
            dz = z_points[i] - z_points[prev_i]
            dt_dist = np.sqrt(dx**2 + dy**2 + dz**2)
            length = length + dt_dist
        print(f"Total length is approximately {length}mm")

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the curve
        ax.plot(x_points, y_points, z_points, 'b-', linewidth=2, label='Clock Spring')
        
        # Mark start and end points
        ax.scatter([x_points[0]], [y_points[0]], [z_points[0]], 
                  color='green', s=100, label='Start', marker='o')
        ax.scatter([x_points[-1]], [y_points[-1]], [z_points[-1]], 
                  color='red', s=100, label='End', marker='s')
        
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
        ax.set_title('Clock Spring Wire Harness Curve')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
            
        return fig


def main():
    """Example usage of the ClockSpringCurve class."""
    
    # Create clock spring curve instance
    clock_spring = ClockSpringCurve()
    
    # Display the parametric equations
    print("Clock Spring Parametric Equations:")
    print("=" * 40)
    x, y, z = clock_spring.get_equations()
    print(f"x(t) = {x}")
    print(f"y(t) = {y}")
    print(f"z(t) = {z}")
    print()
    
    # Example parameter values
    params = {
        clock_spring.r_inner: 40,      # Inner radius: 10 mm
        clock_spring.r_outer: 50,      # Outer radius: 50 mm
        clock_spring.theta_max: 2*sp.pi,  # 2 full rotations
        clock_spring.theta_offset: 0   # No angular offset
    }
    
    print("Example with parameters:")
    print(f"Inner radius: {params[clock_spring.r_inner]} mm")
    print(f"Outer radius: {params[clock_spring.r_outer]} mm")
    print(f"Max angle: {params[clock_spring.theta_max]/(sp.pi)} pi radians")
    print(f"Rotor position: {params[clock_spring.theta_offset]} radians (stator fixed at 0)")
    print("Planar curve (z = 0) for flat cable applications")
    print()
    
    # Generate numerical points
    points = clock_spring.generate_points(params, num_points=1000)
    print(f"Generated {len(points)} points along the curve")
    print("First 5 points [x, y, z]:")
    for i in range(5):
        print(f"  {points[i]}")
    print()
    
    # Plot the 3D curve
    print("Generating 3D plot...")
    clock_spring.plot_curve(params, num_points=1000)


if __name__ == "__main__":
    main()
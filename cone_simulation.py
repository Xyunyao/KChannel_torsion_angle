"""
Cone Diffusion Simulation Utilities

This module provides functions to simulate molecular vector diffusion on cone surfaces.
These simulations can be used for testing, validation, or generating synthetic NMR data.

The key difference from xyz_generator.py is that this module works directly with vectors
(e.g., bond vectors, CSA principal axes) rather than full rotation matrices.
"""

import numpy as np


def rotation_matrix_from_vectors(a, b):
    """
    Find the rotation matrix that aligns vector a to vector b.
    
    Parameters
    ----------
    a : np.ndarray
        Source vector (will be normalized)
    b : np.ndarray
        Target vector (will be normalized)
    
    Returns
    -------
    np.ndarray
        3x3 rotation matrix that rotates a onto b
    
    Notes
    -----
    This uses Rodrigues' rotation formula for the general case.
    Special cases (parallel/antiparallel vectors) are handled separately.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    
    # Already aligned
    if c == 1:
        return np.eye(3)
    
    # Antiparallel - 180° rotation around perpendicular axis
    if c == -1:
        perp = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
        return rotation_matrix_from_vectors(a, np.cross(a, perp))
    
    # General case - Rodrigues' formula
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))


def simulate_vector_on_cone(S2=0.85, tau_c=0.01, dt=1e-4, num_steps=10000, axis=np.array([0, 0, 1])):
    """
    Simulate a unit vector diffusing on a cone surface (fixed θ, azimuthal diffusion only).
    
    This function models diffusion restricted to the cone edge, where the polar angle θ
    is constant and only the azimuthal angle φ changes over time. The azimuthal diffusion
    follows an Ornstein-Uhlenbeck process.
    
    Parameters
    ----------
    S2 : float, optional
        Order parameter (default: 0.85)
        Determines cone angle via: cos(θ) = √((2*S² + 1) / 3)
        This formula assumes the lipari-szabo model with axial symmetry
    tau_c : float, optional
        Correlation time in seconds (default: 0.01 s)
        Defines the timescale of azimuthal diffusion
    dt : float, optional
        Time step in seconds (default: 1e-4 s)
    num_steps : int, optional
        Number of simulation steps (default: 10000)
    axis : np.ndarray, optional
        Cone axis direction (default: [0, 0, 1])
        Will be normalized automatically
    
    Returns
    -------
    vectors : np.ndarray
        Array of unit vectors with shape (num_steps, 3)
        Each row is a unit vector on the cone surface
    
    Notes
    -----
    - The cone angle formula cos(θ) = √((2*S² + 1) / 3) comes from the Lipari-Szabo
      model for axially symmetric motion with fixed cone angle
    - The azimuthal angle φ follows Ornstein-Uhlenbeck dynamics:
      dφ/dt = -γ*φ + σ*ξ(t), where γ = 1/τ_c and σ = √(2γ)
    - This is equivalent to the 'diffusion_cone_edge' model in xyz_generator.py,
      but works directly with vectors instead of rotation matrices
    
    Examples
    --------
    >>> # Simulate NH bond vector diffusion on cone
    >>> vectors = simulate_vector_on_cone(S2=0.85, tau_c=1e-9, dt=1e-12, num_steps=1000)
    >>> print(vectors.shape)
    (1000, 3)
    
    >>> # Calculate actual S² from simulated trajectory
    >>> cos_theta = vectors[:, 2]  # z-component (axis is [0,0,1])
    >>> P2 = (3 * cos_theta**2 - 1) / 2
    >>> S2_calc = np.mean(P2)
    >>> print(f"Target S²: 0.85, Calculated S²: {S2_calc:.4f}")
    
    See Also
    --------
    simulate_vector_within_cone : Simulate diffusion within cone volume (β varies)
    xyz_generator.TrajectoryGenerator : Generate full rotation matrix trajectories
    """
    # Cone angle from S² using Lipari-Szabo formula
    cos_theta = np.sqrt((2 * S2 + 1) / 3)
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    # Ornstein-Uhlenbeck parameters for azimuthal diffusion
    gamma = 1 / tau_c
    sigma = np.sqrt(2 * gamma)  # Unit noise strength
    phi = 0.0
    
    # Normalize cone axis
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix to align cone with specified axis
    R_align = rotation_matrix_from_vectors(np.array([0, 0, 1]), axis)

    vectors = np.zeros((num_steps, 3))

    for i in range(num_steps):
        # Update azimuthal angle using Ornstein-Uhlenbeck process
        dphi = -gamma * phi * dt + sigma * np.sqrt(dt) * np.random.randn()
        phi += dphi

        # Point on cone with fixed θ and current φ
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        vec_local = np.array([x, y, z])

        # Rotate to align cone with specified axis
        vec_global = R_align @ vec_local
        vectors[i] = vec_global

    return vectors


def simulate_vector_within_cone(S2=0.85, tau_c=0.01, dt=1e-4, num_steps=10000, axis=np.array([0, 0, 1])):
    """
    Simulate a unit vector diffusing within a cone volume (both θ and φ vary).
    
    This function models diffusion throughout the cone volume, where both the polar
    angle θ (0 ≤ θ ≤ θ_cone) and azimuthal angle φ change over time.
    
    Parameters
    ----------
    S2 : float, optional
        Order parameter (default: 0.85)
        Determines cone angle via: cos(θ_cone) = 2√S² - 1
    tau_c : float, optional
        Correlation time in seconds (default: 0.01 s)
    dt : float, optional
        Time step in seconds (default: 1e-4 s)
    num_steps : int, optional
        Number of simulation steps (default: 10000)
    axis : np.ndarray, optional
        Cone axis direction (default: [0, 0, 1])
        Will be normalized automatically
    
    Returns
    -------
    vectors : np.ndarray
        Array of unit vectors with shape (num_steps, 3)
    
    Notes
    -----
    - Cone angle formula: cos(θ_cone) = 2√S² - 1
    - Both θ and φ undergo diffusion with correlation time τ_c
    - S² = ((1 + cos(θ_cone))/2)² for this model
    - This corresponds to 'diffusion_cone' in xyz_generator.py
    
    Examples
    --------
    >>> vectors = simulate_vector_within_cone(S2=0.85, tau_c=1e-9, dt=1e-12, num_steps=1000)
    >>> # Verify cone angle constraint
    >>> theta_max = np.arccos(vectors[:, 2]).max()  # Max angle from z-axis
    >>> theta_cone_theory = np.arccos(2*np.sqrt(0.85) - 1)
    >>> print(f"Max theta: {np.degrees(theta_max):.1f}°")
    >>> print(f"Cone angle: {np.degrees(theta_cone_theory):.1f}°")
    """
    # Cone angle from S²
    cos_theta_cone = 2 * np.sqrt(S2) - 1
    theta_cone = np.arccos(np.clip(cos_theta_cone, -1, 1))
    
    # Diffusion parameters
    D_rot = 1 / (6 * tau_c)  # Rotational diffusion coefficient
    sigma_angle = np.sqrt(2 * D_rot * dt)
    
    # Normalize cone axis
    axis = axis / np.linalg.norm(axis)
    R_align = rotation_matrix_from_vectors(np.array([0, 0, 1]), axis)
    
    vectors = np.zeros((num_steps, 3))
    
    # Initialize at cone center
    theta = 0.0
    phi = 0.0
    
    for i in range(num_steps):
        # Diffuse both angles
        d_theta = sigma_angle * np.random.randn()
        d_phi = sigma_angle * np.random.randn()
        
        theta_new = theta + d_theta
        phi_new = phi + d_phi
        
        # Reflect at boundaries (confine within cone)
        if theta_new < 0:
            theta_new = -theta_new
            phi_new += np.pi
        if theta_new > theta_cone:
            theta_new = 2 * theta_cone - theta_new
            phi_new += np.pi
        
        theta = theta_new
        phi = phi_new % (2 * np.pi)
        
        # Convert to Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        vec_local = np.array([x, y, z])
        
        # Rotate to align with specified axis
        vec_global = R_align @ vec_local
        vectors[i] = vec_global
    
    return vectors


def calculate_order_parameter(vectors, axis=np.array([0, 0, 1])):
    """
    Calculate the order parameter S² from a trajectory of vectors.
    
    Parameters
    ----------
    vectors : np.ndarray
        Array of vectors with shape (num_steps, 3)
    axis : np.ndarray, optional
        Reference axis (default: [0, 0, 1])
    
    Returns
    -------
    float
        Order parameter S² = <P₂(cos θ)>
    
    Notes
    -----
    S² is calculated as the time average of the second Legendre polynomial:
    P₂(cos θ) = (3 cos²θ - 1) / 2
    """
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.dot(vectors, axis)
    P2 = (3 * cos_theta**2 - 1) / 2
    return np.mean(P2)


if __name__ == '__main__':
    """Test and demonstration of cone simulation functions."""
    print("="*70)
    print("Cone Simulation Module - Test Suite")
    print("="*70)
    
    # Parameters
    S2_target = 0.85
    tau_c = 1e-9  # 1 ns
    dt = 1e-12    # 1 ps
    num_steps = 5000
    
    print(f"\nSimulation parameters:")
    print(f"  S² target: {S2_target}")
    print(f"  τ_c: {tau_c*1e9:.1f} ns")
    print(f"  dt: {dt*1e12:.1f} ps")
    print(f"  num_steps: {num_steps}")
    print(f"  Total time: {num_steps*dt*1e9:.1f} ns")
    
    # Test 1: Vector on cone edge
    print("\n" + "="*70)
    print("TEST 1: Diffusion on Cone Edge (fixed θ)")
    print("="*70)
    vectors_edge = simulate_vector_on_cone(
        S2=S2_target, 
        tau_c=tau_c, 
        dt=dt, 
        num_steps=num_steps
    )
    
    # Calculate statistics
    cos_theta_edge = vectors_edge[:, 2]  # z-component
    theta_edge = np.arccos(cos_theta_edge)
    
    print(f"  Generated {len(vectors_edge)} vectors")
    print(f"  Theta (polar angle):")
    print(f"    Mean:  {np.degrees(theta_edge.mean()):.4f}°")
    print(f"    Std:   {np.degrees(theta_edge.std()):.6f}°")
    print(f"    Range: {np.degrees(theta_edge.min()):.4f}° to {np.degrees(theta_edge.max()):.4f}°")
    
    # Calculate S²
    S2_calc_edge = calculate_order_parameter(vectors_edge)
    cos_theta_theory = np.sqrt((2 * S2_target + 1) / 3)
    theta_theory = np.arccos(cos_theta_theory)
    
    print(f"\n  S² validation:")
    print(f"    Target S²:     {S2_target:.6f}")
    print(f"    Calculated S²: {S2_calc_edge:.6f}")
    print(f"    Error:         {abs(S2_calc_edge - S2_target):.6f}")
    print(f"\n  Cone angle:")
    print(f"    Theoretical: {np.degrees(theta_theory):.4f}°")
    print(f"    From data:   {np.degrees(theta_edge.mean()):.4f}°")
    
    # Test 2: Vector within cone
    print("\n" + "="*70)
    print("TEST 2: Diffusion Within Cone (θ varies)")
    print("="*70)
    vectors_within = simulate_vector_within_cone(
        S2=S2_target,
        tau_c=tau_c,
        dt=dt,
        num_steps=num_steps
    )
    
    # Calculate statistics
    cos_theta_within = vectors_within[:, 2]
    theta_within = np.arccos(np.clip(cos_theta_within, -1, 1))
    
    print(f"  Generated {len(vectors_within)} vectors")
    print(f"  Theta (polar angle):")
    print(f"    Mean:  {np.degrees(theta_within.mean()):.4f}°")
    print(f"    Std:   {np.degrees(theta_within.std()):.4f}°")
    print(f"    Range: {np.degrees(theta_within.min()):.4f}° to {np.degrees(theta_within.max()):.4f}°")
    
    # Calculate S²
    S2_calc_within = calculate_order_parameter(vectors_within)
    cos_theta_cone = 2 * np.sqrt(S2_target) - 1
    theta_cone = np.arccos(cos_theta_cone)
    
    print(f"\n  S² validation:")
    print(f"    Target S²:     {S2_target:.6f}")
    print(f"    Calculated S²: {S2_calc_within:.6f}")
    print(f"    Error:         {abs(S2_calc_within - S2_target):.6f}")
    print(f"\n  Cone angle:")
    print(f"    Theoretical: {np.degrees(theta_cone):.4f}°")
    print(f"    Max from data: {np.degrees(theta_within.max()):.4f}°")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Edge model:   θ = {np.degrees(theta_theory):.2f}° (fixed), S² = {S2_calc_edge:.4f}")
    print(f"Within model: θ ≤ {np.degrees(theta_cone):.2f}° (varies), S² = {S2_calc_within:.4f}")
    print(f"\nCone angle ratio (edge/within): {theta_theory/theta_cone:.3f}")
    print(f"For same S² = {S2_target}, edge model has smaller cone angle!")
    print("="*70)

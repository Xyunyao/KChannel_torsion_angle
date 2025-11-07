"""
Module 1: XYZ Trajectory Generator

Generate molecular trajectories for NMR parameter calculations.

Supported trajectory types:
1. Diffusion within a cone (Lipari-Szabo model) - 'diffusion_cone'
   - Motion restricted within cone volume (0 ≤ β ≤ θ_cone)
   - S² = ((1 + cos(θ))/2)²
   
2. Diffusion on cone edge/surface - 'diffusion_cone_edge'
   - Motion restricted to cone surface (β = θ_cone, fixed)
   - Azimuthal diffusion only
   - S² = ((1 + cos(θ)) × cos(θ) / 2)²
   
3. Custom trajectory (user-defined function) - 'custom'
4. Load from file (MD trajectory) - 'from_file'
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Callable, List, Tuple

# Handle both package import and standalone execution
try:
    from .config import NMRConfig
except ImportError:
    from config import NMRConfig


class TrajectoryGenerator:
    """
    Base class for trajectory generation.
    
    Attributes
    ----------
    config : NMRConfig
        Configuration object with simulation parameters
    """
    
    def __init__(self, config: NMRConfig):
        """
        Initialize trajectory generator.
        
        Parameters
        ----------
        config : NMRConfig
            Configuration object
        """
        self.config = config
        self.rotations = None
        self.coordinates = None
        
    def generate(self) -> Tuple[List[R], Optional[np.ndarray]]:
        """
        Generate trajectory based on configuration.
        
        Returns
        -------
        rotations : List[scipy.spatial.transform.Rotation]
            List of rotation objects at each time step
        coordinates : np.ndarray or None
            XYZ coordinates if applicable (n_steps, 3)
        """
        if self.config.trajectory_type == 'diffusion_cone':
            return self.generate_diffusion_cone()
        elif self.config.trajectory_type == 'diffusion_cone_edge':
            return self.generate_diffusion_cone_edge()
        elif self.config.trajectory_type == 'custom':
            return self.generate_custom()
        elif self.config.trajectory_type == 'from_file':
            return self.load_from_file()
        else:
            raise ValueError(f"Unknown trajectory type: {self.config.trajectory_type}")
    
    def generate_diffusion_cone(self) -> Tuple[List[R], None]:
        """
        Generate diffusion on a cone trajectory (Lipari-Szabo model).
        
        The molecule diffuses rotationally within a cone, characterized by:
        - Order parameter S²: restricts cone angle
        - Correlation time τc: time scale of diffusion
        
        Returns
        -------
        rotations : List[Rotation]
            List of rotation matrices at each time step
        coordinates : None
            No Cartesian coordinates for this model
        """
        S2 = self.config.S2
        tau_c = self.config.tau_c
        dt = self.config.dt
        num_steps = self.config.num_steps
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 1: Generating Diffusion on Cone Trajectory")
            print(f"{'='*70}")
            print(f"  S² = {S2:.4f}")
            print(f"  τc = {tau_c*1e9:.2f} ns")
            print(f"  dt = {dt*1e12:.2f} ps")
            print(f"  Steps = {num_steps}")
            print(f"  Duration = {num_steps*dt*1e9:.2f} ns")
        
        # Calculate cone semi-angle from S2
        theta_cone = self._S2_to_cone_angle(S2)
        
        if self.config.verbose:
            print(f"  Cone half-angle = {np.degrees(theta_cone):.2f}°")
        
        # Rotational diffusion coefficient
        D_rot = 1.0 / (6.0 * tau_c)
        
        # Angular step size (sqrt of variance for Brownian motion)
        sigma_angle = np.sqrt(2.0 * D_rot * dt)
        
        # Initialize rotation
        phi_0 = np.random.uniform(0, 2*np.pi)
        theta_0 = np.random.uniform(0, theta_cone)
        gamma_0 = np.random.uniform(0, 2*np.pi)
        
        current_rotation = R.from_euler('ZYZ', [phi_0, theta_0, gamma_0])
        rotations = [current_rotation]
        
        # Generate trajectory
        for step in range(num_steps - 1):
            if self.config.verbose and step % 1000 == 0 and step > 0:
                print(f"  Progress: {step}/{num_steps} ({step/num_steps*100:.1f}%)", end='\r')
            
            # Current Euler angles
            euler = current_rotation.as_euler('ZYZ')
            alpha, beta, gamma = euler
            
            # Random walk in beta (polar angle) - restricted to cone
            d_beta = np.random.normal(0, sigma_angle)
            beta_new = beta + d_beta
            
            # Reflect at cone boundary
            if beta_new > theta_cone:
                beta_new = 2*theta_cone - beta_new
            if beta_new < 0:
                beta_new = -beta_new
                alpha = alpha + np.pi  # Flip azimuthal angle
            
            # Random walk in alpha (azimuthal angle) - free diffusion
            d_alpha = np.random.normal(0, sigma_angle / np.sin(beta + 1e-10))
            alpha_new = (alpha + d_alpha) % (2 * np.pi)
            
            # Gamma angle - intrinsic rotation
            d_gamma = np.random.normal(0, sigma_angle)
            gamma_new = (gamma + d_gamma) % (2 * np.pi)
            
            # Create new rotation
            current_rotation = R.from_euler('ZYZ', [alpha_new, beta_new, gamma_new])
            rotations.append(current_rotation)
        
        if self.config.verbose:
            print(f"\n  ✓ Generated {len(rotations)} rotation matrices")
        
        self.rotations = rotations
        return rotations, None
    
    def generate_diffusion_cone_edge(self) -> Tuple[List[R], None]:
        """
        Generate diffusion on the edge (surface) of a cone.
        
        Unlike diffusion within a cone, this model restricts motion to a fixed
        polar angle θ from the cone axis. The molecule diffuses only in the 
        azimuthal direction around the cone surface.
        
        This corresponds to motion on a cone surface at constant angle:
        - β (polar angle) = θ_cone (fixed)
        - α (azimuthal angle) = free diffusion around cone axis
        - γ (intrinsic rotation) = free diffusion
        
        The order parameter S² for diffusion on cone edge is:
        S² = (cos²(θ_cone) + cos(θ_cone))² / 4
        
        Or equivalently:
        S² = ((1 + cos(θ_cone)) × cos(θ_cone) / 2)²
        
        Returns
        -------
        rotations : List[Rotation]
            List of rotation matrices at each time step
        coordinates : None
            No Cartesian coordinates for this model
        """
        S2 = self.config.S2
        tau_c = self.config.tau_c
        dt = self.config.dt
        num_steps = self.config.num_steps
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 1: Generating Diffusion on Cone Edge Trajectory")
            print(f"{'='*70}")
            print(f"  Model: Motion constrained to cone surface")
            print(f"  S² = {S2:.4f}")
            print(f"  τc = {tau_c*1e9:.2f} ns")
            print(f"  dt = {dt*1e12:.2f} ps")
            print(f"  Steps = {num_steps}")
            print(f"  Duration = {num_steps*dt*1e9:.2f} ns")
        
        # Calculate fixed cone angle from S2
        # For cone edge: S² = ((1 + cos(θ)) × cos(θ) / 2)²
        # Solving: let x = cos(θ), then S² = ((1+x)×x/2)²
        # √S² = (1+x)×x/2
        # 2√S² = x + x²
        # x² + x - 2√S² = 0
        # x = (-1 + √(1 + 8√S²)) / 2
        
        sqrt_S2 = np.sqrt(S2)
        cos_theta_cone = (-1 + np.sqrt(1 + 8*sqrt_S2)) / 2
        cos_theta_cone = np.clip(cos_theta_cone, -1, 1)
        theta_cone = np.arccos(cos_theta_cone)
        
        if self.config.verbose:
            print(f"  Fixed cone angle θ = {np.degrees(theta_cone):.2f}°")
            print(f"  cos(θ) = {cos_theta_cone:.4f}")
        
        # Rotational diffusion coefficient
        D_rot = 1.0 / (6.0 * tau_c)
        
        # Angular step size for azimuthal diffusion
        # On cone surface, effective diffusion is faster in azimuthal direction
        sigma_angle = np.sqrt(2.0 * D_rot * dt)
        
        # For azimuthal angle, diffusion is scaled by 1/sin(θ)
        sigma_alpha = sigma_angle / (np.sin(theta_cone) + 1e-10)
        sigma_gamma = sigma_angle  # Intrinsic rotation
        
        if self.config.verbose:
            print(f"  σ_α (azimuthal) = {np.degrees(sigma_alpha):.3f}°")
            print(f"  σ_γ (intrinsic) = {np.degrees(sigma_gamma):.3f}°")
        
        # Initialize rotation
        alpha_0 = np.random.uniform(0, 2*np.pi)
        beta_0 = theta_cone  # Fixed at cone angle
        gamma_0 = np.random.uniform(0, 2*np.pi)
        
        current_rotation = R.from_euler('ZYZ', [alpha_0, beta_0, gamma_0])
        rotations = [current_rotation]
        
        # Generate trajectory - diffusion on cone surface
        for step in range(num_steps - 1):
            if self.config.verbose and step % 1000 == 0 and step > 0:
                print(f"  Progress: {step}/{num_steps} ({step/num_steps*100:.1f}%)", end='\r')
            
            # Current Euler angles
            euler = current_rotation.as_euler('ZYZ')
            alpha, beta, gamma = euler
            
            # Beta stays fixed at cone angle (key difference from within-cone diffusion)
            beta_new = theta_cone
            
            # Azimuthal diffusion around cone axis
            d_alpha = np.random.normal(0, sigma_alpha)
            alpha_new = (alpha + d_alpha) % (2 * np.pi)
            
            # Intrinsic rotation (gamma angle)
            d_gamma = np.random.normal(0, sigma_gamma)
            gamma_new = (gamma + d_gamma) % (2 * np.pi)
            
            # Create new rotation
            current_rotation = R.from_euler('ZYZ', [alpha_new, beta_new, gamma_new])
            rotations.append(current_rotation)
        
        if self.config.verbose:
            print(f"\n  ✓ Generated {len(rotations)} rotation matrices")
            print(f"  ✓ All rotations at fixed β = {np.degrees(theta_cone):.2f}°")
        
        self.rotations = rotations
        return rotations, None
    
    def simulate_vector_on_cone(self, 
                                S2: Optional[float] = None,
                                tau_c: Optional[float] = None,
                                dt: Optional[float] = None,
                                num_steps: Optional[int] = None,
                                axis: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate a unit vector diffusing on a cone surface (fixed θ, azimuthal diffusion only).
        
        This function models diffusion restricted to the cone edge, where the polar angle θ
        is constant and only the azimuthal angle φ changes over time. The azimuthal diffusion
        follows an Ornstein-Uhlenbeck process.
        
        Parameters
        ----------
        S2 : float, optional
            Order parameter (default: use config.S2)
            Determines cone angle via: cos(θ) = √((2*S² + 1) / 3)
        tau_c : float, optional
            Correlation time in seconds (default: use config.tau_c)
        dt : float, optional
            Time step in seconds (default: use config.dt)
        num_steps : int, optional
            Number of simulation steps (default: use config.num_steps)
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
        - This generates vectors (not full rotation matrices) for simple applications
        
        Examples
        --------
        >>> config = NMRConfig(S2=0.85, tau_c=1e-9, dt=1e-12, num_steps=1000)
        >>> gen = TrajectoryGenerator(config)
        >>> vectors = gen.simulate_vector_on_cone()
        >>> print(vectors.shape)
        (1000, 3)
        
        >>> # With custom axis
        >>> vectors = gen.simulate_vector_on_cone(axis=np.array([1, 0, 0]))
        """
        # Use config values as defaults
        S2 = S2 if S2 is not None else self.config.S2
        tau_c = tau_c if tau_c is not None else self.config.tau_c
        dt = dt if dt is not None else self.config.dt
        num_steps = num_steps if num_steps is not None else self.config.num_steps
        axis = axis if axis is not None else np.array([0, 0, 1])
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 1: Simulating Vector on Cone Surface")
            print(f"{'='*70}")
            print(f"  S² = {S2:.4f}")
            print(f"  τc = {tau_c*1e9:.2f} ns")
            print(f"  dt = {dt*1e12:.2f} ps")
            print(f"  Steps = {num_steps}")
            print(f"  Cone axis = {axis}")
        
        # Cone angle from S² using Lipari-Szabo formula
        cos_theta = np.sqrt((2 * S2 + 1) / 3)
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        
        if self.config.verbose:
            print(f"  Cone angle θ = {np.degrees(theta):.2f}°")
        
        # Ornstein-Uhlenbeck parameters for azimuthal diffusion
        gamma = 1 / tau_c
        sigma = np.sqrt(2 * gamma)  # Unit noise strength
        phi = 0.0
        
        # Normalize cone axis
        axis = axis / np.linalg.norm(axis)
        
        # Rotation matrix to align cone with specified axis
        R_align = self._rotation_matrix_from_vectors(np.array([0, 0, 1]), axis)
        
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
        
        if self.config.verbose:
            print(f"\n  ✓ Generated {len(vectors)} unit vectors")
            print(f"  ✓ All vectors at fixed angle θ = {np.degrees(theta):.2f}° from axis")
        
        return vectors
    
    @staticmethod
    def _rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
        Uses Rodrigues' rotation formula for the general case.
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
            return TrajectoryGenerator._rotation_matrix_from_vectors(a, np.cross(a, perp))
        
        # General case - Rodrigues' formula
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    
    def generate_custom(self, 
                       custom_function: Optional[Callable] = None) -> Tuple[List[R], Optional[np.ndarray]]:
        """
        Generate trajectory using custom user-defined function.
        
        Parameters
        ----------
        custom_function : Callable, optional
            Function that generates trajectory.
            Signature: custom_function(config) -> (rotations, coordinates)
        
        Returns
        -------
        rotations : List[Rotation]
            List of rotation matrices
        coordinates : np.ndarray or None
            XYZ coordinates if provided
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 1: Generating Custom Trajectory")
            print(f"{'='*70}")
        
        if custom_function is None:
            # Placeholder: Use simple random walk as example
            if self.config.verbose:
                print("  Warning: No custom function provided. Using random walk placeholder.")
            return self._generate_random_walk_placeholder()
        
        rotations, coordinates = custom_function(self.config)
        
        if self.config.verbose:
            print(f"  ✓ Generated {len(rotations)} frames")
            if coordinates is not None:
                print(f"  ✓ Coordinates shape: {coordinates.shape}")
        
        self.rotations = rotations
        self.coordinates = coordinates
        return rotations, coordinates
    
    def load_from_file(self, filepath: Optional[str] = None) -> Tuple[List[R], np.ndarray]:
        """
        Load trajectory from file.
        
        Parameters
        ----------
        filepath : str, optional
            Path to trajectory file (XYZ, PDB, or NPZ format)
        
        Returns
        -------
        rotations : List[Rotation]
            List of rotation matrices (computed from coordinates)
        coordinates : np.ndarray
            XYZ coordinates (n_steps, n_atoms, 3)
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 1: Loading Trajectory from File")
            print(f"{'='*70}")
        
        if filepath is None:
            raise ValueError("filepath must be provided for 'from_file' trajectory type")
        
        if self.config.verbose:
            print(f"  Loading: {filepath}")
        
        # Placeholder: Add file loading logic here
        # Support formats: .xyz, .pdb, .npz
        if filepath.endswith('.xyz'):
            coordinates = self._load_xyz(filepath)
        elif filepath.endswith('.npz'):
            coordinates = self._load_npz(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        # Placeholder: Convert coordinates to rotations
        # This requires defining local axis from coordinates
        rotations = None  # Will be computed in Module 2
        
        if self.config.verbose:
            print(f"  ✓ Loaded {len(coordinates)} frames")
        
        self.coordinates = coordinates
        self.rotations = rotations
        return rotations, coordinates
    
    def _S2_to_cone_angle(self, S2: float) -> float:
        """
        Convert order parameter S² to cone half-angle.
        
        For diffusion on a cone:
        S² ≈ ((1 + cos(θ))/2)²  (simplified model)
        
        Parameters
        ----------
        S2 : float
            Order parameter (0 < S2 < 1)
        
        Returns
        -------
        theta_cone : float
            Cone half-angle in radians
        """
        if S2 >= 1.0:
            return 0.0  # No motion
        elif S2 <= 0.0:
            return np.pi / 2  # Isotropic
        else:
            # S² = ((1 + cos(θ))/2)²
            # sqrt(S²) = (1 + cos(θ))/2
            # cos(θ) = 2*sqrt(S²) - 1
            cos_theta = 2 * np.sqrt(S2) - 1
            cos_theta = np.clip(cos_theta, -1, 1)
            return np.arccos(cos_theta)
    
    def _generate_random_walk_placeholder(self) -> Tuple[List[R], None]:
        """
        Placeholder: Generate simple random walk trajectory.
        
        Returns
        -------
        rotations : List[Rotation]
            Random rotations
        coordinates : None
        """
        rotations = []
        for i in range(self.config.num_steps):
            # Random rotation
            random_angles = np.random.uniform(0, 2*np.pi, 3)
            rot = R.from_euler('ZYZ', random_angles)
            rotations.append(rot)
        
        return rotations, None
    
    def _load_xyz(self, filepath: str) -> np.ndarray:
        """
        Placeholder: Load XYZ file.
        
        Parameters
        ----------
        filepath : str
            Path to XYZ file
        
        Returns
        -------
        coordinates : np.ndarray
            Coordinates (n_steps, n_atoms, 3)
        """
        # Placeholder implementation
        # TODO: Implement actual XYZ file parsing
        raise NotImplementedError("XYZ file loading not yet implemented. "
                                "Please implement _load_xyz() method.")
    
    def _load_npz(self, filepath: str) -> np.ndarray:
        """
        Load NPZ file containing coordinates or rotation matrices.
        
        Parameters
        ----------
        filepath : str
            Path to NPZ file
        
        Returns
        -------
        coordinates : np.ndarray
            Coordinates or rotation matrices
        """
        data = np.load(filepath)
        
        # Try to find coordinates in common key names
        for key in ['coordinates', 'coords', 'xyz', 'positions']:
            if key in data:
                return data[key]
        
        # If not found, use first array
        keys = list(data.keys())
        if len(keys) > 0:
            if self.config.verbose:
                print(f"  Using key: {keys[0]}")
            return data[keys[0]]
        
        raise ValueError(f"No coordinate data found in {filepath}")
    
    def save(self, filepath: str):
        """
        Save generated trajectory.
        
        Parameters
        ----------
        filepath : str
            Output file path (.npz format)
        """
        save_dict = {}
        
        if self.rotations is not None:
            # Save rotations as quaternions
            quats = np.array([r.as_quat() for r in self.rotations])
            save_dict['quaternions'] = quats
            
            # Also save as Euler angles
            eulers = np.array([r.as_euler('ZYZ') for r in self.rotations])
            save_dict['euler_angles'] = eulers
        
        if self.coordinates is not None:
            save_dict['coordinates'] = self.coordinates
        
        # Save configuration
        save_dict['config'] = str(self.config.to_dict())
        
        np.savez(filepath, **save_dict)
        
        if self.config.verbose:
            print(f"  ✓ Saved trajectory to: {filepath}")


# Example usage and testing
if __name__ == '__main__':
    print("="*70)
    print("TESTING MODULE 1: XYZ TRAJECTORY GENERATOR")
    print("="*70)
    
    # Test 1: Diffusion within cone
    print("\n" + "="*70)
    print("TEST 1: Diffusion WITHIN Cone")
    print("="*70)
    config_within = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        dt=2e-12,
        num_steps=1000,
        verbose=True
    )
    
    generator_within = TrajectoryGenerator(config_within)
    rotations_within, _ = generator_within.generate()
    
    # Analyze beta angles
    betas_within = np.array([r.as_euler('ZYZ')[1] for r in rotations_within])
    print(f"\n  Beta angle statistics (within cone):")
    print(f"    Mean: {np.degrees(np.mean(betas_within)):.2f}°")
    print(f"    Std:  {np.degrees(np.std(betas_within)):.2f}°")
    print(f"    Min:  {np.degrees(np.min(betas_within)):.2f}°")
    print(f"    Max:  {np.degrees(np.max(betas_within)):.2f}°")
    
    # Test 2: Diffusion on cone edge
    print("\n" + "="*70)
    print("TEST 2: Diffusion ON Cone Edge (Surface)")
    print("="*70)
    config_edge = NMRConfig(
        trajectory_type='diffusion_cone_edge',
        S2=0.85,
        tau_c=2e-9,
        dt=2e-12,
        num_steps=1000,
        verbose=True
    )
    
    generator_edge = TrajectoryGenerator(config_edge)
    rotations_edge, _ = generator_edge.generate()
    
    # Analyze beta angles
    betas_edge = np.array([r.as_euler('ZYZ')[1] for r in rotations_edge])
    print(f"\n  Beta angle statistics (cone edge):")
    print(f"    Mean: {np.degrees(np.mean(betas_edge)):.2f}°")
    print(f"    Std:  {np.degrees(np.std(betas_edge)):.2f}°")
    print(f"    Min:  {np.degrees(np.min(betas_edge)):.2f}°")
    print(f"    Max:  {np.degrees(np.max(betas_edge)):.2f}°")
    print(f"    Range: {np.degrees(np.max(betas_edge) - np.min(betas_edge)):.4f}°")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"  Within cone - Beta varies: {np.degrees(betas_within.min()):.2f}° to {np.degrees(betas_within.max()):.2f}°")
    print(f"  On edge     - Beta fixed:  {np.degrees(betas_edge.mean()):.2f}° ± {np.degrees(betas_edge.std()):.4f}°")
    print(f"\n  Within cone explores volume (0 ≤ β ≤ θ_cone)")
    print(f"  On edge stays on surface (β = θ_cone, constant)")
    
    # Test 3: Compare S² calculation
    print("\n" + "="*70)
    print("TEST 3: S² Validation")
    print("="*70)
    
    # Calculate S² from trajectory
    def calculate_S2_from_trajectory(rotations):
        """Calculate S² from P2(cos(β)) average."""
        betas = np.array([r.as_euler('ZYZ')[1] for r in rotations])
        cos_beta = np.cos(betas)
        P2_cos_beta = (3 * cos_beta**2 - 1) / 2
        S2_calc = np.mean(P2_cos_beta)
        return S2_calc
    
    S2_within_calc = calculate_S2_from_trajectory(rotations_within)
    S2_edge_calc = calculate_S2_from_trajectory(rotations_edge)
    
    print(f"  Within cone:")
    print(f"    Input S²:      {config_within.S2:.4f}")
    print(f"    Calculated S²: {S2_within_calc:.4f}")
    print(f"    Error:         {abs(S2_within_calc - config_within.S2):.4f}")
    
    print(f"\n  On edge:")
    print(f"    Input S²:      {config_edge.S2:.4f}")
    print(f"    Calculated S²: {S2_edge_calc:.4f}")
    print(f"    Error:         {abs(S2_edge_calc - config_edge.S2):.4f}")
    
    # Test 4: simulate_vector_on_cone method
    print("\n" + "="*70)
    print("TEST 4: simulate_vector_on_cone() Method")
    print("="*70)
    
    print("\n  4a. Default axis [0, 0, 1]:")
    config_vec = NMRConfig(
        S2=0.85,
        tau_c=1e-9,
        dt=1e-12,
        num_steps=500,
        verbose=False
    )
    gen_vec = TrajectoryGenerator(config_vec)
    vectors_z = gen_vec.simulate_vector_on_cone()
    
    # Calculate angle from z-axis
    theta_z = np.arccos(np.clip(vectors_z[:, 2], -1, 1))
    print(f"    Generated {len(vectors_z)} vectors")
    print(f"    Angle from z-axis:")
    print(f"      Mean: {np.degrees(theta_z.mean()):.4f}°")
    print(f"      Std:  {np.degrees(theta_z.std()):.6f}°")
    
    print("\n  4b. Custom axis [1, 0, 0] (x-axis):")
    vectors_x = gen_vec.simulate_vector_on_cone(axis=np.array([1, 0, 0]))
    
    # Calculate angle from x-axis
    theta_x = np.arccos(np.clip(vectors_x[:, 0], -1, 1))
    print(f"    Generated {len(vectors_x)} vectors")
    print(f"    Angle from x-axis:")
    print(f"      Mean: {np.degrees(theta_x.mean()):.4f}°")
    print(f"      Std:  {np.degrees(theta_x.std()):.6f}°")
    
    print("\n  4c. Custom axis [0, 1, 1] (diagonal):")
    vectors_diag = gen_vec.simulate_vector_on_cone(axis=np.array([0, 1, 1]))
    diag_axis = np.array([0, 1, 1]) / np.linalg.norm([0, 1, 1])
    
    # Calculate angle from diagonal axis
    theta_diag = np.arccos(np.clip(np.dot(vectors_diag, diag_axis), -1, 1))
    print(f"    Generated {len(vectors_diag)} vectors")
    print(f"    Angle from diagonal axis:")
    print(f"      Mean: {np.degrees(theta_diag.mean()):.4f}°")
    print(f"      Std:  {np.degrees(theta_diag.std()):.6f}°")
    
    # Validate S² from vectors
    def calculate_S2_from_vectors(vectors, axis):
        """Calculate S² from vectors."""
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.dot(vectors, axis)
        P2 = (3 * cos_theta**2 - 1) / 2
        return np.mean(P2)
    
    S2_vec_z = calculate_S2_from_vectors(vectors_z, np.array([0, 0, 1]))
    S2_vec_x = calculate_S2_from_vectors(vectors_x, np.array([1, 0, 0]))
    S2_vec_diag = calculate_S2_from_vectors(vectors_diag, diag_axis)
    
    print(f"\n  S² validation:")
    print(f"    Target S²:         {config_vec.S2:.6f}")
    print(f"    From z-axis vecs:  {S2_vec_z:.6f} (error: {abs(S2_vec_z - config_vec.S2):.6f})")
    print(f"    From x-axis vecs:  {S2_vec_x:.6f} (error: {abs(S2_vec_x - config_vec.S2):.6f})")
    print(f"    From diag vecs:    {S2_vec_diag:.6f} (error: {abs(S2_vec_diag - config_vec.S2):.6f})")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)

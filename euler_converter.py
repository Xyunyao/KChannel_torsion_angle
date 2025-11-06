"""
Module 2: Euler Angle Converter

Convert XYZ coordinates or rotation matrices to Euler angles based on local axis definitions.

Supported input modes:
1. From Module 1 rotations (rotation matrices)
2. From coordinates (extract local axes)
3. From file (load Euler angles directly) - BYPASSES MODULE 1

Supported local axis definitions:
1. CO-Cα (backbone carbonyl-alpha carbon) - default
2. N-H (backbone amide)
3. Custom (user-defined vectors)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Optional, Callable, Union
from pathlib import Path

# Handle both package import and standalone execution
try:
    from .config import NMRConfig
except ImportError:
    from config import NMRConfig


class EulerConverter:
    """
    Convert molecular coordinates or rotations to Euler angles in lab frame.
    
    Attributes
    ----------
    config : NMRConfig
        Configuration object
    """
    
    def __init__(self, config: NMRConfig):
        """
        Initialize Euler angle converter.
        
        Parameters
        ----------
        config : NMRConfig
            Configuration object with local axis definition
        """
        self.config = config
        self.euler_angles = None
        
    def convert(self, 
                rotations: Optional[List[R]] = None,
                coordinates: Optional[np.ndarray] = None,
                from_file: Optional[str] = None) -> np.ndarray:
        """
        Convert rotations or coordinates to Euler angles, or load from file.
        
        Three input modes:
        1. rotations: Pre-computed rotation matrices from Module 1
        2. coordinates: Extract local axes from molecular coordinates
        3. from_file: Load Euler angles directly (BYPASSES MODULE 1)
        
        Parameters
        ----------
        rotations : List[Rotation], optional
            List of rotation objects from Module 1
        coordinates : np.ndarray, optional
            Coordinates (n_steps, n_atoms, 3) for extracting local axes
        from_file : str, optional
            Path to file containing Euler angles
            Supported formats: .npz, .npy, .txt, .csv
        
        Returns
        -------
        euler_angles : np.ndarray
            Euler angles in ZYZ convention (n_steps, 3)
            Columns: [alpha, beta, gamma] in radians
        
        Notes
        -----
        If from_file is provided, Module 1 is completely bypassed.
        This is useful for:
        - Loading pre-computed orientations from MD simulations
        - Using external trajectory analysis tools
        - Skipping trajectory generation entirely
        """
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("MODULE 2: Converting to Euler Angles")
            print(f"{'='*70}")
        
        # Priority order: from_file > rotations > coordinates
        if from_file is not None:
            euler_angles = self.load_euler_from_file(from_file)
            if self.config.verbose:
                print(f"  ✓ Loaded {len(euler_angles)} Euler angle sets from file")
                print(f"  ✓ MODULE 1 BYPASSED - using pre-computed orientations")
        elif rotations is not None:
            if self.config.verbose:
                print(f"  Local axis definition: {self.config.local_axis_definition}")
            euler_angles = self._rotations_to_euler(rotations)
        elif coordinates is not None:
            if self.config.verbose:
                print(f"  Local axis definition: {self.config.local_axis_definition}")
            euler_angles = self._coordinates_to_euler(coordinates)
        else:
            raise ValueError("Must provide one of: rotations, coordinates, or from_file")
        
        if self.config.verbose:
            print(f"  Shape: {euler_angles.shape}")
            print(f"  Convention: ZYZ (alpha, beta, gamma)")
            print(f"  Angular ranges:")
            print(f"    α: {np.degrees(euler_angles[:, 0].min()):.1f}° to {np.degrees(euler_angles[:, 0].max()):.1f}°")
            print(f"    β: {np.degrees(euler_angles[:, 1].min()):.1f}° to {np.degrees(euler_angles[:, 1].max()):.1f}°")
            print(f"    γ: {np.degrees(euler_angles[:, 2].min()):.1f}° to {np.degrees(euler_angles[:, 2].max()):.1f}°")
        
        self.euler_angles = euler_angles
        return euler_angles
    
    def _rotations_to_euler(self, rotations: List[R]) -> np.ndarray:
        """
        Extract Euler angles from rotation objects.
        
        Parameters
        ----------
        rotations : List[Rotation]
            List of scipy Rotation objects
        
        Returns
        -------
        euler_angles : np.ndarray
            Euler angles (n_steps, 3) in ZYZ convention
        """
        euler_angles = np.array([rot.as_euler('ZYZ') for rot in rotations])
        return euler_angles
    
    def load_euler_from_file(self, filepath: str) -> np.ndarray:
        """
        Load Euler angles directly from file.
        
        This method allows bypassing Module 1 entirely by loading pre-computed
        orientations from MD simulations or other sources.
        
        Parameters
        ----------
        filepath : str
            Path to file containing Euler angles
            Supported formats:
            - .npz: NumPy compressed archive (keys: 'euler_angles', 'alpha_beta_gamma', 'angles')
            - .npy: NumPy binary array
            - .txt, .dat, .csv: Text files (3 columns: alpha, beta, gamma)
        
        Returns
        -------
        euler_angles : np.ndarray
            Euler angles (n_steps, 3) in radians [alpha, beta, gamma]
            If input is in degrees, will be converted to radians
        
        Notes
        -----
        Expected format:
        - Shape: (n_steps, 3) where columns are [α, β, γ]
        - Convention: ZYZ Euler angles
        - Units: Radians (or degrees with auto-detection)
        
        File format examples:
        
        **NPZ format** (recommended):
        ```python
        np.savez('euler_angles.npz', 
                 euler_angles=angles,  # (n_steps, 3) in radians
                 convention='ZYZ',
                 units='radians')
        ```
        
        **Text format**:
        ```
        # alpha beta gamma (radians)
        0.1234 0.5678 0.9012
        0.2345 0.6789 1.0123
        ...
        ```
        
        Examples
        --------
        >>> # Load from NPZ file
        >>> converter = EulerConverter(config)
        >>> euler = converter.load_euler_from_file('trajectory_euler.npz')
        
        >>> # Load from text file
        >>> euler = converter.load_euler_from_file('euler_angles.txt')
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if self.config.verbose:
            print(f"  Loading Euler angles from: {filepath}")
            print(f"  File format: {filepath.suffix}")
        
        # Load based on file extension
        if filepath.suffix == '.npz':
            euler_angles = self._load_euler_npz(filepath)
        elif filepath.suffix == '.npy':
            euler_angles = self._load_euler_npy(filepath)
        elif filepath.suffix in ['.txt', '.dat', '.csv']:
            euler_angles = self._load_euler_text(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}. "
                           f"Supported: .npz, .npy, .txt, .dat, .csv")
        
        # Validate shape
        if euler_angles.ndim != 2 or euler_angles.shape[1] != 3:
            raise ValueError(f"Invalid Euler angle array shape: {euler_angles.shape}. "
                           f"Expected (n_steps, 3)")
        
        # Auto-detect and convert degrees to radians
        euler_angles = self._ensure_radians(euler_angles)
        
        if self.config.verbose:
            print(f"  ✓ Loaded {len(euler_angles)} Euler angle sets")
            print(f"  ✓ Units: radians")
        
        return euler_angles
    
    def _load_euler_npz(self, filepath: Path) -> np.ndarray:
        """Load Euler angles from NPZ file."""
        data = np.load(filepath, allow_pickle=True)
        
        # Try common key names
        for key in ['euler_angles', 'angles', 'alpha_beta_gamma', 'orientations', 'eulers']:
            if key in data:
                if self.config.verbose:
                    print(f"    Using key: '{key}'")
                return data[key]
        
        # If no recognized key, use first array
        keys = [k for k in data.keys() if not k.startswith('_')]
        if len(keys) > 0:
            if self.config.verbose:
                print(f"    Using first available key: '{keys[0]}'")
                print(f"    Available keys: {keys}")
            return data[keys[0]]
        
        raise ValueError(f"No Euler angle data found in {filepath}")
    
    def _load_euler_npy(self, filepath: Path) -> np.ndarray:
        """Load Euler angles from NPY file."""
        return np.load(filepath)
    
    def _load_euler_text(self, filepath: Path) -> np.ndarray:
        """Load Euler angles from text file."""
        try:
            # Try loading with numpy (handles comments, whitespace)
            euler_angles = np.loadtxt(filepath)
            return euler_angles
        except Exception as e:
            raise ValueError(f"Failed to load text file {filepath}: {e}")
    
    def _ensure_radians(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Auto-detect if angles are in degrees and convert to radians.
        
        Heuristic: If any angle is > 2π (≈6.28), assume degrees.
        """
        max_angle = np.abs(euler_angles).max()
        
        if max_angle > 2 * np.pi:
            # Likely in degrees
            if self.config.verbose:
                print(f"    Auto-detected degrees (max value: {np.degrees(max_angle):.1f}°)")
                print(f"    Converting to radians...")
            return np.radians(euler_angles)
        else:
            if self.config.verbose:
                print(f"    Angles already in radians (max value: {max_angle:.3f} rad = {np.degrees(max_angle):.1f}°)")
            return euler_angles
    
    def _coordinates_to_euler(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Extract Euler angles from molecular coordinates using local axis definition.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Molecular coordinates (n_steps, n_atoms, 3)
        
        Returns
        -------
        euler_angles : np.ndarray
            Euler angles (n_steps, 3) in ZYZ convention
        """
        if self.config.local_axis_definition == 'CO_CA':
            return self._extract_euler_CO_CA(coordinates)
        elif self.config.local_axis_definition == 'NH':
            return self._extract_euler_NH(coordinates)
        elif self.config.local_axis_definition == 'custom':
            return self._extract_euler_custom(coordinates)
        else:
            raise ValueError(f"Unknown local axis definition: {self.config.local_axis_definition}")
    
    def _extract_euler_CO_CA(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Extract Euler angles using CO-Cα local axis definition (backbone).
        
        Local axis frame:
        - z-axis: along C=O bond (carbonyl)
        - x-axis: in plane of Cα-C-O, perpendicular to z
        - y-axis: completes right-handed system
        
        Parameters
        ----------
        coordinates : np.ndarray
            Coordinates (n_steps, n_atoms, 3)
            Expected atom order: [..., N, Cα, C, O, ...]
        
        Returns
        -------
        euler_angles : np.ndarray
            Euler angles (n_steps, 3)
        """
        if self.config.verbose:
            print("  Using CO-Cα backbone axis definition")
            print("    z-axis: C=O bond direction")
            print("    x-axis: perpendicular in Cα-C-O plane")
        
        n_steps = coordinates.shape[0]
        euler_angles = np.zeros((n_steps, 3))
        
        # Placeholder: Need to specify atom indices
        # For now, assume standard protein backbone ordering
        # This is where user would specify which atoms to use
        
        for i in range(n_steps):
            if self.config.verbose and i % 1000 == 0 and i > 0:
                print(f"    Progress: {i}/{n_steps} ({i/n_steps*100:.1f}%)", end='\r')
            
            # PLACEHOLDER: Extract CO and CA positions
            # User should provide atom indices or atom names
            # For example:
            # C_pos = coordinates[i, C_index]
            # O_pos = coordinates[i, O_index]
            # CA_pos = coordinates[i, CA_index]
            
            # For now, create dummy local frame
            # In real use, replace with actual coordinate extraction
            z_axis = np.array([0, 0, 1])  # Placeholder
            x_axis = np.array([1, 0, 0])  # Placeholder
            y_axis = np.cross(z_axis, x_axis)
            
            # Create rotation matrix from local axes to lab frame
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
            # Convert to Euler angles
            rot = R.from_matrix(rotation_matrix)
            euler_angles[i] = rot.as_euler('ZYZ')
        
        if self.config.verbose:
            print("\n    ⚠️  Using placeholder implementation - specify atom indices for real data!")
        
        return euler_angles
    
    def _extract_euler_NH(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Extract Euler angles using N-H local axis definition (backbone amide).
        
        Local axis frame:
        - z-axis: along N-H bond
        - x-axis: perpendicular in N-C-H plane
        - y-axis: completes right-handed system
        
        Parameters
        ----------
        coordinates : np.ndarray
            Coordinates (n_steps, n_atoms, 3)
        
        Returns
        -------
        euler_angles : np.ndarray
            Euler angles (n_steps, 3)
        """
        if self.config.verbose:
            print("  Using N-H backbone axis definition")
            print("    z-axis: N-H bond direction")
        
        n_steps = coordinates.shape[0]
        euler_angles = np.zeros((n_steps, 3))
        
        # PLACEHOLDER: Similar to CO-CA, need atom indices
        # User should specify N, H, and C (for plane definition)
        
        for i in range(n_steps):
            # Placeholder local frame
            z_axis = np.array([0, 0, 1])
            x_axis = np.array([1, 0, 0])
            y_axis = np.cross(z_axis, x_axis)
            
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            rot = R.from_matrix(rotation_matrix)
            euler_angles[i] = rot.as_euler('ZYZ')
        
        if self.config.verbose:
            print("    ⚠️  Using placeholder - specify N, H atom indices!")
        
        return euler_angles
    
    def _extract_euler_custom(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Extract Euler angles using custom local axis definition.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Coordinates (n_steps, n_atoms, 3)
        
        Returns
        -------
        euler_angles : np.ndarray
            Euler angles (n_steps, 3)
        """
        if self.config.verbose:
            print("  Using custom axis definition")
        
        if self.config.custom_axis_vectors is None:
            raise ValueError("custom_axis_vectors must be provided in config for custom axis definition")
        
        # PLACEHOLDER: User provides function to extract axes
        # Example format:
        # custom_axis_vectors = (extract_z_axis_func, extract_x_axis_func)
        # where each function takes coordinates[i] and returns a vector
        
        n_steps = coordinates.shape[0]
        euler_angles = np.zeros((n_steps, 3))
        
        if self.config.verbose:
            print("    ⚠️  Implement custom axis extraction function!")
        
        return euler_angles
    
    @staticmethod
    def create_rotation_from_axes(z_axis: np.ndarray, 
                                   x_axis: Optional[np.ndarray] = None,
                                   reference_plane: Optional[np.ndarray] = None) -> R:
        """
        Create rotation matrix from local axes.
        
        Parameters
        ----------
        z_axis : np.ndarray
            Z-axis direction (will be normalized)
        x_axis : np.ndarray, optional
            X-axis direction (will be orthogonalized to z)
        reference_plane : np.ndarray, optional
            Vector in xy-plane for defining x-axis
        
        Returns
        -------
        rotation : Rotation
            Rotation object representing local-to-lab frame transformation
        """
        # Normalize z-axis
        z = z_axis / np.linalg.norm(z_axis)
        
        if x_axis is not None:
            # Orthogonalize x to z
            x = x_axis - np.dot(x_axis, z) * z
            x = x / np.linalg.norm(x)
        elif reference_plane is not None:
            # Create x in reference plane, perpendicular to z
            x = reference_plane - np.dot(reference_plane, z) * z
            x = x / np.linalg.norm(x)
        else:
            # Choose arbitrary x perpendicular to z
            if abs(z[2]) < 0.9:
                x = np.cross(z, np.array([0, 0, 1]))
            else:
                x = np.cross(z, np.array([1, 0, 0]))
            x = x / np.linalg.norm(x)
        
        # Y-axis completes right-handed system
        y = np.cross(z, x)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack([x, y, z])
        
        return R.from_matrix(rotation_matrix)
    
    @staticmethod
    def extract_CO_CA_axes(C_pos: np.ndarray, 
                          O_pos: np.ndarray, 
                          CA_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract local axes from CO-Cα coordinates.
        
        Parameters
        ----------
        C_pos : np.ndarray
            Carbonyl carbon position (3,)
        O_pos : np.ndarray
            Carbonyl oxygen position (3,)
        CA_pos : np.ndarray
            Alpha carbon position (3,)
        
        Returns
        -------
        z_axis : np.ndarray
            Z-axis along C=O
        x_axis : np.ndarray
            X-axis in Cα-C-O plane
        y_axis : np.ndarray
            Y-axis completing right-handed system
        """
        # Z-axis along C=O
        z_axis = O_pos - C_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # X-axis in Cα-C-O plane, perpendicular to z
        ca_c_vector = C_pos - CA_pos
        x_axis = ca_c_vector - np.dot(ca_c_vector, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis completes right-handed system
        y_axis = np.cross(z_axis, x_axis)
        
        return z_axis, x_axis, y_axis
    
    @staticmethod
    def extract_NH_axes(N_pos: np.ndarray, 
                       H_pos: np.ndarray,
                       C_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract local axes from N-H coordinates.
        
        Parameters
        ----------
        N_pos : np.ndarray
            Nitrogen position (3,)
        H_pos : np.ndarray
            Hydrogen position (3,)
        C_pos : np.ndarray
            Carbon position (for defining plane) (3,)
        
        Returns
        -------
        z_axis : np.ndarray
            Z-axis along N-H
        x_axis : np.ndarray
            X-axis in C-N-H plane
        y_axis : np.ndarray
            Y-axis completing right-handed system
        """
        # Z-axis along N-H
        z_axis = H_pos - N_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # X-axis in C-N-H plane
        c_n_vector = N_pos - C_pos
        x_axis = c_n_vector - np.dot(c_n_vector, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis
        y_axis = np.cross(z_axis, x_axis)
        
        return z_axis, x_axis, y_axis
    
    def save(self, filepath: str):
        """
        Save Euler angles.
        
        Parameters
        ----------
        filepath : str
            Output file path (.npz format)
        """
        if self.euler_angles is None:
            raise ValueError("No Euler angles to save. Run convert() first.")
        
        np.savez(filepath, 
                euler_angles=self.euler_angles,
                config=str(self.config.to_dict()))
        
        if self.config.verbose:
            print(f"  ✓ Saved Euler angles to: {filepath}")


# Example usage
if __name__ == '__main__':
    try:
        from config import NMRConfig
        from xyz_generator import TrajectoryGenerator
    except ImportError:
        print("Importing from package...")
        from .config import NMRConfig
        from .xyz_generator import TrajectoryGenerator
    
    import tempfile
    
    print("="*70)
    print("TESTING MODULE 2: EULER ANGLE CONVERTER")
    print("="*70)
    
    # Test 1: Convert from Module 1 rotations
    print("\n" + "="*70)
    print("TEST 1: Convert from Module 1 Rotations")
    print("="*70)
    
    config = NMRConfig(
        trajectory_type='diffusion_cone',
        S2=0.85,
        tau_c=2e-9,
        num_steps=100,
        verbose=True
    )
    
    gen = TrajectoryGenerator(config)
    rotations, _ = gen.generate()
    
    converter = EulerConverter(config)
    euler_angles = converter.convert(rotations=rotations)
    
    print(f"\n  Euler angles shape: {euler_angles.shape}")
    print(f"  First 3 sets:\n{np.degrees(euler_angles[:3])}")
    
    # Test 2: Save and reload Euler angles (bypassing Module 1)
    print("\n" + "="*70)
    print("TEST 2: Save and Reload Euler Angles (Bypass Module 1)")
    print("="*70)
    
    # Create temporary files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save in different formats
        npz_file = tmpdir / "euler_angles.npz"
        npy_file = tmpdir / "euler_angles.npy"
        txt_file = tmpdir / "euler_angles.txt"
        
        # Save as NPZ (recommended)
        np.savez(npz_file, 
                euler_angles=euler_angles,
                convention='ZYZ',
                units='radians')
        print(f"\n  Saved to NPZ: {npz_file.name}")
        
        # Save as NPY
        np.save(npy_file, euler_angles)
        print(f"  Saved to NPY: {npy_file.name}")
        
        # Save as text (in degrees for readability)
        np.savetxt(txt_file, np.degrees(euler_angles), 
                  header='alpha beta gamma (degrees)',
                  fmt='%.6f')
        print(f"  Saved to TXT: {txt_file.name}")
        
        # Test loading from NPZ (MODULE 1 BYPASSED)
        print("\n  2a. Loading from NPZ (bypassing Module 1):")
        config_load = NMRConfig(verbose=True)
        converter_npz = EulerConverter(config_load)
        euler_npz = converter_npz.convert(from_file=str(npz_file))
        
        # Verify data matches
        diff_npz = np.abs(euler_npz - euler_angles).max()
        print(f"     Max difference from original: {diff_npz:.2e} rad")
        print(f"     ✓ Match: {np.allclose(euler_npz, euler_angles)}")
        
        # Test loading from NPY
        print("\n  2b. Loading from NPY (bypassing Module 1):")
        converter_npy = EulerConverter(config_load)
        euler_npy = converter_npy.convert(from_file=str(npy_file))
        
        diff_npy = np.abs(euler_npy - euler_angles).max()
        print(f"     Max difference from original: {diff_npy:.2e} rad")
        print(f"     ✓ Match: {np.allclose(euler_npy, euler_angles)}")
        
        # Test loading from TXT (in degrees, should auto-convert)
        print("\n  2c. Loading from TXT (bypassing Module 1, auto-convert degrees→radians):")
        converter_txt = EulerConverter(config_load)
        euler_txt = converter_txt.convert(from_file=str(txt_file))
        
        diff_txt = np.abs(euler_txt - euler_angles).max()
        print(f"     Max difference from original: {diff_txt:.2e} rad")
        print(f"     ✓ Match: {np.allclose(euler_txt, euler_angles, rtol=1e-5)}")
    
    # Test 3: Create test file with known angles
    print("\n" + "="*70)
    print("TEST 3: Verify Angle Ranges")
    print("="*70)
    
    print(f"\n  Alpha (α) range:")
    print(f"    Min: {np.degrees(euler_angles[:, 0].min()):7.2f}°")
    print(f"    Max: {np.degrees(euler_angles[:, 0].max()):7.2f}°")
    print(f"    Mean: {np.degrees(euler_angles[:, 0].mean()):7.2f}°")
    
    print(f"\n  Beta (β) range:")
    print(f"    Min: {np.degrees(euler_angles[:, 1].min()):7.2f}°")
    print(f"    Max: {np.degrees(euler_angles[:, 1].max()):7.2f}°")
    print(f"    Mean: {np.degrees(euler_angles[:, 1].mean()):7.2f}°")
    
    print(f"\n  Gamma (γ) range:")
    print(f"    Min: {np.degrees(euler_angles[:, 2].min()):7.2f}°")
    print(f"    Max: {np.degrees(euler_angles[:, 2].max()):7.2f}°")
    print(f"    Mean: {np.degrees(euler_angles[:, 2].mean()):7.2f}°")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
    print("\n✓ Module 2 can now load Euler angles directly from files")
    print("✓ This bypasses Module 1 entirely")
    print("✓ Useful for pre-computed MD trajectories")

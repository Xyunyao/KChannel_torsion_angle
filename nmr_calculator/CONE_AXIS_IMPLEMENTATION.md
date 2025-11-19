# Cone Axis Parameter Implementation

## Summary

Added support for customizable cone axis direction in the `vector_on_cone` trajectory generation. All config parameters (S2, tau_c, dt, num_steps, cone_axis) are now properly passed through the method chain.

## Changes Made

### 1. `config.py` (NMRConfig class)
- **Added parameter**: `cone_axis: Optional[np.ndarray] = None`
  - Default: `None` (uses `[0, 0, 1]` as z-axis)
  - Allows specifying arbitrary cone axis direction
  - Will be automatically normalized
  
- **Added documentation** in class docstring

### 2. `xyz_generator.py` (TrajectoryGenerator class)

#### `generate_vector_on_cone_trajectory()` method
- **Return type changed**: `Tuple[List[R], None]` → `Tuple[List[R], np.ndarray]`
- **Now returns**: Both rotation matrices AND unit vectors
- **Explicitly passes all config parameters** to `simulate_vector_on_cone()`:
  ```python
  vectors = self.simulate_vector_on_cone(
      S2=self.config.S2,
      tau_c=self.config.tau_c,
      dt=self.config.dt,
      num_steps=self.config.num_steps,
      axis=axis
  )
  ```
- **Stores vectors**: Sets `self.coordinates = vectors` for downstream use

#### `simulate_vector_on_cone()` method
- **Updated default axis logic**: 
  ```python
  if axis is None:
      axis = self.config.cone_axis if self.config.cone_axis is not None else np.array([0, 0, 1])
  ```
- **All parameters have config defaults**: S2, tau_c, dt, num_steps, and axis all fall back to config values
- **Ensures array type**: Converts input axis to numpy array

## Usage Examples

### Example 1: Default Z-axis
```python
config = NMRConfig(
    trajectory_type='vector_on_cone',
    S2=0.85,
    tau_c=2e-10,
    dt=1e-12,
    num_steps=1000
)
gen = TrajectoryGenerator(config)
rotations, vectors = gen.generate()
# Vectors will be on cone around [0, 0, 1]
```

### Example 2: Custom X-axis
```python
config = NMRConfig(
    trajectory_type='vector_on_cone',
    S2=0.85,
    tau_c=2e-10,
    dt=1e-12,
    num_steps=1000,
    cone_axis=np.array([1.0, 0.0, 0.0])  # X-axis
)
gen = TrajectoryGenerator(config)
rotations, vectors = gen.generate()
# Vectors will be on cone around [1, 0, 0]
```

### Example 3: Arbitrary Axis
```python
custom_axis = np.array([1.0, 1.0, 1.0])  # Diagonal direction
config = NMRConfig(
    trajectory_type='vector_on_cone',
    S2=0.85,
    tau_c=2e-10,
    dt=1e-12,
    num_steps=1000,
    cone_axis=custom_axis
)
gen = TrajectoryGenerator(config)
rotations, vectors = gen.generate()
# Vectors will be on cone around normalized [1, 1, 1]
```

## Validation

Test script: `test_cone_axis.py`

Results show:
- ✅ Default axis (z-axis) works correctly
- ✅ Custom axis (x-axis) works correctly  
- ✅ Arbitrary axis (diagonal) works correctly
- ✅ All vectors maintain constant angle from specified axis (18.43° for S²=0.85)
- ✅ Both rotations and vectors are returned

## Technical Details

### Cone Angle Calculation
The cone angle θ is calculated from the order parameter S² using the Lipari-Szabo formula:
```
cos(θ) = √((2*S² + 1) / 3)
```

For S² = 0.85: θ = 18.43°

### Axis Transformation
Vectors are generated in a local coordinate system (z-aligned cone), then rotated to align with the specified axis using a rotation matrix calculated by `_rotation_matrix_from_vectors()`.

### Output Format
- **rotations**: List of scipy `Rotation` objects (for Euler angle conversion)
- **vectors**: numpy array of shape `(num_steps, 3)` containing unit vectors on cone surface

## Backward Compatibility

✅ Fully backward compatible:
- If `cone_axis` is not specified, defaults to z-axis `[0, 0, 1]`
- Existing code will continue to work without modification

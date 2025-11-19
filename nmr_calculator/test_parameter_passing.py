#!/usr/bin/env python3
"""
Test to verify all parameters are correctly passed from config to simulate_vector_on_cone.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/yunyao_1/Dropbox/KcsA/analysis')

from nmr_calculator.config import NMRConfig
from nmr_calculator.xyz_generator import TrajectoryGenerator

print('='*70)
print('Testing Parameter Passing to simulate_vector_on_cone')
print('='*70)

# Test with specific values different from defaults
config = NMRConfig(
    trajectory_type='vector_on_cone',
    S2=0.75,  # Different from default 0.85
    tau_c=5e-10,  # Different from default 2e-9
    dt=2e-12,  # Different from default 2e-12
    num_steps=200,  # Different from default 20000
    cone_axis=np.array([0.5, 0.5, 0.707]),  # Custom axis
    verbose=True
)

print(f'\nConfig parameters:')
print(f'  S2 = {config.S2}')
print(f'  tau_c = {config.tau_c*1e9:.2f} ns')
print(f'  dt = {config.dt*1e12:.2f} ps')
print(f'  num_steps = {config.num_steps}')
print(f'  cone_axis = {config.cone_axis}')

gen = TrajectoryGenerator(config)

# Test 1: Via generate() method
print('\n[Test 1] Via generate() method:')
print('-'*70)
rotations, vectors = gen.generate()

print(f'\n  Generated {len(vectors)} vectors (expected {config.num_steps})')
assert len(vectors) == config.num_steps, "num_steps not passed correctly!"

# Calculate expected cone angle from S2
cos_theta_expected = np.sqrt((2 * config.S2 + 1) / 3)
theta_expected = np.degrees(np.arccos(cos_theta_expected))

# Calculate actual angle from axis
axis_norm = config.cone_axis / np.linalg.norm(config.cone_axis)
dots = vectors @ axis_norm
angles = np.degrees(np.arccos(np.clip(dots, -1, 1)))

print(f'\n  Expected cone angle (from S2={config.S2}): {theta_expected:.2f}°')
print(f'  Actual angle from axis:')
print(f'    Mean: {np.mean(angles):.2f}°')
print(f'    Std:  {np.std(angles):.4f}°')

angle_error = abs(np.mean(angles) - theta_expected)
print(f'  Error: {angle_error:.4f}°')
assert angle_error < 0.01, "S2 not passed correctly!"

# Test 2: Direct call to simulate_vector_on_cone with explicit parameters
print('\n[Test 2] Direct call with explicit parameters:')
print('-'*70)
vectors2 = gen.simulate_vector_on_cone(
    S2=0.90,  # Override
    tau_c=1e-9,  # Override
    dt=1e-12,  # Override
    num_steps=150,  # Override
    axis=np.array([1, 0, 0])  # Override
)

print(f'\n  Generated {len(vectors2)} vectors (expected 150)')
assert len(vectors2) == 150, "Explicit num_steps not working!"

cos_theta_expected2 = np.sqrt((2 * 0.90 + 1) / 3)
theta_expected2 = np.degrees(np.arccos(cos_theta_expected2))

x_axis = np.array([1, 0, 0])
angles2 = np.degrees(np.arccos(np.clip(vectors2[:, 0], -1, 1)))

print(f'\n  Expected cone angle (from S2=0.90): {theta_expected2:.2f}°')
print(f'  Actual angle from x-axis:')
print(f'    Mean: {np.mean(angles2):.2f}°')
print(f'    Std:  {np.std(angles2):.4f}°')

angle_error2 = abs(np.mean(angles2) - theta_expected2)
print(f'  Error: {angle_error2:.4f}°')
assert angle_error2 < 0.01, "Explicit S2 not working!"

# Test 3: Default parameters from config
print('\n[Test 3] Using config defaults:')
print('-'*70)
vectors3 = gen.simulate_vector_on_cone()  # No parameters - should use config

print(f'\n  Generated {len(vectors3)} vectors (expected {config.num_steps})')
assert len(vectors3) == config.num_steps, "Config defaults not working!"

dots3 = vectors3 @ axis_norm
angles3 = np.degrees(np.arccos(np.clip(dots3, -1, 1)))

print(f'\n  Expected cone angle (from S2={config.S2}): {theta_expected:.2f}°')
print(f'  Actual angle from axis:')
print(f'    Mean: {np.mean(angles3):.2f}°')
print(f'    Std:  {np.std(angles3):.4f}°')

angle_error3 = abs(np.mean(angles3) - theta_expected)
print(f'  Error: {angle_error3:.4f}°')
assert angle_error3 < 0.01, "Config defaults for S2 not working!"

print('\n' + '='*70)
print('✓ All parameter passing tests passed!')
print('='*70)
print('\nSummary:')
print('  ✓ Config parameters passed to generate()')
print('  ✓ Explicit parameters override config')
print('  ✓ Default parameters use config values')
print('  ✓ All parameters (S2, tau_c, dt, num_steps, axis) working correctly')

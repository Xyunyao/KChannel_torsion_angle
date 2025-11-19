#!/usr/bin/env python3
"""
Test script for cone_axis parameter in vector_on_cone trajectory generation.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/yunyao_1/Dropbox/KcsA/analysis')

from nmr_calculator.config import NMRConfig
from nmr_calculator.xyz_generator import TrajectoryGenerator

print('='*70)
print('Testing cone_axis Parameter for vector_on_cone Trajectory')
print('='*70)

# Test 1: Default axis (z-axis)
print('\n[Test 1] Default axis (should be [0, 0, 1])')
print('-'*70)
config1 = NMRConfig(
    trajectory_type='vector_on_cone',
    S2=0.85,
    tau_c=2e-10,
    dt=1e-12,
    num_steps=100,
    verbose=True
)

gen1 = TrajectoryGenerator(config1)
rotations1, vectors1 = gen1.generate()

print(f'\nReturned vectors shape: {vectors1.shape}')
print(f'First 3 vectors:')
for i in range(3):
    print(f'  [{vectors1[i,0]:7.4f}, {vectors1[i,1]:7.4f}, {vectors1[i,2]:7.4f}]')

# Calculate angle from z-axis
angles_from_z = np.arccos(np.clip(vectors1[:, 2], -1, 1))
print(f'\nAngle from z-axis:')
print(f'  Mean: {np.degrees(np.mean(angles_from_z)):.2f}°')
print(f'  Std:  {np.degrees(np.std(angles_from_z)):.2f}°')

# Test 2: Custom axis (x-axis)
print('\n[Test 2] Custom axis [1, 0, 0] (x-axis)')
print('-'*70)
config2 = NMRConfig(
    trajectory_type='vector_on_cone',
    S2=0.85,
    tau_c=2e-10,
    dt=1e-12,
    num_steps=100,
    cone_axis=np.array([1.0, 0.0, 0.0]),
    verbose=True
)

gen2 = TrajectoryGenerator(config2)
rotations2, vectors2 = gen2.generate()

print(f'\nReturned vectors shape: {vectors2.shape}')
print(f'First 3 vectors:')
for i in range(3):
    print(f'  [{vectors2[i,0]:7.4f}, {vectors2[i,1]:7.4f}, {vectors2[i,2]:7.4f}]')

# Calculate angle from x-axis
x_axis = np.array([1, 0, 0])
dots = vectors2 @ x_axis
angles_from_x = np.arccos(np.clip(dots, -1, 1))
print(f'\nAngle from x-axis:')
print(f'  Mean: {np.degrees(np.mean(angles_from_x)):.2f}°')
print(f'  Std:  {np.degrees(np.std(angles_from_x)):.2f}°')

# Test 3: Custom axis (arbitrary direction)
print('\n[Test 3] Custom axis [1, 1, 1] (diagonal)')
print('-'*70)
custom_axis = np.array([1.0, 1.0, 1.0])
config3 = NMRConfig(
    trajectory_type='vector_on_cone',
    S2=0.85,
    tau_c=2e-10,
    dt=1e-12,
    num_steps=100,
    cone_axis=custom_axis,
    verbose=True
)

gen3 = TrajectoryGenerator(config3)
rotations3, vectors3 = gen3.generate()

print(f'\nReturned vectors shape: {vectors3.shape}')
print(f'First 3 vectors:')
for i in range(3):
    print(f'  [{vectors3[i,0]:7.4f}, {vectors3[i,1]:7.4f}, {vectors3[i,2]:7.4f}]')

# Calculate angle from custom axis
custom_axis_norm = custom_axis / np.linalg.norm(custom_axis)
dots = vectors3 @ custom_axis_norm
angles_from_custom = np.arccos(np.clip(dots, -1, 1))
print(f'\nAngle from custom axis [{custom_axis_norm[0]:.3f}, {custom_axis_norm[1]:.3f}, {custom_axis_norm[2]:.3f}]:')
print(f'  Mean: {np.degrees(np.mean(angles_from_custom)):.2f}°')
print(f'  Std:  {np.degrees(np.std(angles_from_custom)):.2f}°')

# Expected cone angle from S2
S2 = 0.85
cos_theta = np.sqrt((2 * S2 + 1) / 3)
theta_expected = np.degrees(np.arccos(cos_theta))
print(f'\nExpected cone angle from S²={S2}: {theta_expected:.2f}°')

print('\n' + '='*70)
print('✓ All tests completed successfully!')
print('='*70)

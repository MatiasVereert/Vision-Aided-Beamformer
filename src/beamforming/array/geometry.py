import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_log_array_coords(M: int, d_min: float, d_max: float, room_dims: np.ndarray) -> np.ndarray:
    """
    Generates a 1D logarithmic (geometric) microphone array centered in the room.
    The array is aligned along the X-axis.
    """
    if M < 2:
        raise ValueError("At least 2 microphones are required.")
    if d_min >= d_max:
        raise ValueError("d_min must be strictly less than d_max.")
        
    if M == 2:
        pos_x = np.array([0.0, d_max])
    else:
        pos_x = np.zeros(M)
        pos_x[1:] = np.geomspace(d_min, d_max, num=M-1)
        
    pos_x = pos_x - (d_max / 2.0)
    
    mic_coords = np.zeros((M, 3))
    mic_coords[:, 0] = pos_x
    
    room_center = np.array(room_dims) / 2.0
    mic_coords = mic_coords + room_center
    
    margin = 0.1 
    if np.any(mic_coords < margin) or np.any(mic_coords > (np.array(room_dims) - margin)):
        raise ValueError(f"The array of length {d_max}m does not fit safely.")
        
    return mic_coords


def generate_source_and_interferences(N_interferences: int, radius_source: float, radius_interf: float, delta_ang_deg: float, array_center: np.ndarray) -> tuple:
    """
    Generates the 3D coordinates for the target source and N interferences.
    The target source is fixed at broadside (90 degrees, perpendicular to X-axis array).
    Interferences are placed alternately at +/- multiples of delta_ang relative to the source.
    """
    delta_ang_rad = np.deg2rad(delta_ang_deg)
    
    # The array is on the X-axis. Broadside (perpendicular) is the Y-axis (90 degrees or pi/2)
    ref_angle_rad = np.pi / 2.0
    
    # Calculate target source position
    source_pos = np.copy(array_center)
    source_pos[0] += radius_source * np.cos(ref_angle_rad) # Evaluates to ~0 offset in X
    source_pos[1] += radius_source * np.sin(ref_angle_rad) # Evaluates to radius_source in Y
    
    interferences_pos = np.zeros((N_interferences, 3))
    
    for i in range(N_interferences):
        multiplier = (i // 2) + 1
        sign = 1 if i % 2 == 0 else -1
        
        # Calculate angle relative to the broadside reference
        angle_rad = ref_angle_rad + (sign * multiplier * delta_ang_rad)
        
        # Calculate cartesian coordinates
        x = array_center[0] + radius_interf * np.cos(angle_rad)
        y = array_center[1] + radius_interf * np.sin(angle_rad)
        z = array_center[2]
        
        interferences_pos[i] = [x, y, z]
        
    return source_pos, interferences_pos


if __name__ == "__main__":
    # 1. Define room and array setup parameters
    room_dims = np.array([6.0, 5.0, 2.5])
    M = 8
    d_min = 0.02
    d_max = 0.30
    
    # Source and interference setup parameters
    radius_source = 1.0   # Fuente más cerca del arreglo
    radius_interf = 1.8   # Interferencias más alejadas
    delta_ang_deg = 30.0  # Espaciado angular
    
    # 2. Compute array coordinates and center
    mic_coords = generate_log_array_coords(M, d_min, d_max, room_dims)
    array_center = room_dims / 2.0
    
    # 3. Create the Matplotlib figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    interference_cases = [1, 2, 3]
    
    for ax, n_int in zip(axes, interference_cases):
        # Generate sources for the current case using the updated function
        source_pos, interferences_pos = generate_source_and_interferences(
            N_interferences=n_int, 
            radius_source=radius_source,
            radius_interf=radius_interf, 
            delta_ang_deg=delta_ang_deg, 
            array_center=array_center
        )
        
        # Plot room boundary (Rectangle)
        room_patch = patches.Rectangle(
            (0, 0), room_dims[0], room_dims[1], 
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
        )
        ax.add_patch(room_patch)
        
        # Plot the microphone array
        ax.scatter(mic_coords[:, 0], mic_coords[:, 1], c='blue', marker='x', label='Mic Array')
        
        # Plot the target source (Green)
        ax.scatter(source_pos[0], source_pos[1], c='green', marker='o', s=100, label='Target Source')
        
        # Plot the interferences and add their tags
        for i in range(n_int):
            ax.scatter(interferences_pos[i, 0], interferences_pos[i, 1], c='red', marker='v', s=80)
            
            # Tag text positioning (slightly offset from the point)
            offset_x = 0.1
            offset_y = 0.1
            ax.text(
                interferences_pos[i, 0] + offset_x, 
                interferences_pos[i, 1] + offset_y, 
                f'Int {i+1}', 
                fontsize=10, color='red', weight='bold'
            )
            
        # Plot formatting
        ax.set_xlim(-0.5, room_dims[0] + 0.5)
        ax.set_ylim(-0.5, room_dims[1] + 0.5)
        ax.set_aspect('equal') # Keep physical proportions true
        ax.set_title(f'{n_int} Interference(s)')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, linestyle=':', alpha=0.7)
        
        if n_int == 1:
            ax.legend(loc='upper left')
            
    plt.tight_layout()
    plt.show()
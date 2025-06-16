import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree
import psutil
import time
import tqdm
from numba import cuda
import warnings
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.mass = np.float32(mass)

@cuda.jit
def compute_density_kernel(pair_indices, positions, masses, densities, args):
    idx = cuda.grid(1)
    h, k, mu, rho_0 = args
    if idx < pair_indices.shape[0] // 2:
        i = pair_indices[2 * idx]
        j = pair_indices[2 * idx + 1]

        pos_i = positions[i]
        pos_j = positions[j]

        dx = pos_j[0] - pos_i[0]
        dy = pos_j[1] - pos_i[1]
        r = math.sqrt(dx * dx + dy * dy)

        if r < h:
            W = 315.0 * (h**2 - r**2)**3 / (64.0 * math.pi * h**9)
        else:
            W = 0.0

        cuda.atomic.add(densities, i, masses[j] * W)
        cuda.atomic.add(densities, j, masses[i] * W)

@cuda.jit(device=True)
def pair_accel(pos_i, pos_j, vel_i, vel_j, dens_i, dens_j, mass_i, mass_j, time, args):
    h, k, mu, rho_0 = args

    dx = pos_j[0] - pos_i[0]
    dy = pos_j[1] - pos_i[1]
    r2 = dx**2 + dy**2
    h2 = h**2

    if r2 > h2 or r2 == 0:
        return (0.0, 0.0), (0.0, 0.0)

    r = math.sqrt(r2)
    rx, ry = dx, dy

    poly6_grad_coeff = -945.0 / (32.0 * math.pi * h**9)
    poly6_lap_coeff  =  945.0 / (8.0  * math.pi * h**9)

    hr2 = h2 - r2
    gradW = poly6_grad_coeff * hr2**2
    lapW = poly6_lap_coeff * hr2 * (3.0 * r2 - h2)

    gradWx = gradW * rx if r != 0 else 0.0
    gradWy = gradW * ry if r != 0 else 0.0

    min_rho = 1e0  # minimum density to avoid div-by-zero
    safe_dens_i = max(dens_i, min_rho)
    safe_dens_j = max(dens_j, min_rho)

    p_i = k * (safe_dens_i - rho_0)
    p_j = k * (safe_dens_j - rho_0)

    denom_i = safe_dens_i * safe_dens_i
    denom_j = safe_dens_j * safe_dens_j
    pressure_term = - (p_i / denom_i + p_j / denom_j)

    fx_p = pressure_term * gradWx * mass_j
    fy_p = pressure_term * gradWy * mass_j

    vel_diff_x = vel_j[0] - vel_i[0]
    vel_diff_y = vel_j[1] - vel_i[1]
    fx_v = mu * vel_diff_x * lapW * mass_j / dens_j
    fy_v = mu * vel_diff_y * lapW * mass_j / dens_j

    fx_total = fx_p + fx_v
    fy_total = fy_p + fy_v

    return (fx_total / mass_i, fy_total / mass_i), (-fx_total / mass_j, -fy_total / mass_j)

@cuda.jit
def compute_all_pair_forces(pair_indices, velocities, positions, density, masses, accels, time, args):
    idx = cuda.grid(1)
    if idx < pair_indices.shape[0] // 2:
        i = pair_indices[2 * idx]
        j = pair_indices[2 * idx + 1]

        pos_i = positions[i]
        pos_j = positions[j]
        vel_i = velocities[i]
        vel_j = velocities[j]
        mass_i = masses[i]
        mass_j = masses[j]
        dens_i = density[i]
        dens_j = density[j]

        (ax_i, ay_i), (ax_j, ay_j) = pair_accel(pos_i, pos_j, vel_i, vel_j, dens_i, dens_j, mass_i, mass_j, time, args)
        
        max_accel = 1e4  # adjust as needed
        ax_i = max(-max_accel, min(ax_i, max_accel))
        ay_i = max(-max_accel, min(ay_i, max_accel))
        ax_j = max(-max_accel, min(ax_j, max_accel))
        ay_j = max(-max_accel, min(ay_j, max_accel))
        
        cuda.atomic.add(accels, (i, 0), ax_i)
        cuda.atomic.add(accels, (i, 1), ay_i)
        cuda.atomic.add(accels, (j, 0), ax_j)
        cuda.atomic.add(accels, (j, 1), ay_j)

def compute_accels(time, positions, velocities, masses, args):
    h = args['h']
    k = args['k']
    mu = args['mu']
    rho_0 = args['rho_0']
    container_size = args['container_size']
    params = np.array([h, k, mu, rho_0], dtype=np.float32)
    r_cut = h
    n = len(positions)
    accels = np.zeros_like(positions, dtype=np.float32)
    tree = cKDTree(positions)
    pairs = np.array(list(tree.query_pairs(r=r_cut)), dtype=np.int32).flatten()

    assert len(pairs) % 2 == 0, "pair_indices must be a flat array of shape (2*N,)"

    num_pairs = len(pairs) // 2
    if num_pairs == 0:
        return accels

    gravity = np.array([0, -10], dtype=np.float32)
    accels += gravity

    stream = cuda.stream()
    d_positions = cuda.to_device(positions.astype(np.float32), stream=stream)
    d_velocities = cuda.to_device(velocities.astype(np.float32), stream=stream)
    d_masses = cuda.to_device(masses.astype(np.float32), stream=stream)
    d_accels = cuda.to_device(accels.astype(np.float32), stream=stream)
    d_pairs = cuda.to_device(pairs, stream=stream)

    d_densities = cuda.device_array(n, dtype=np.float32, stream=stream)
    d_densities[:] = 0

    threadsperblock = 128
    blockspergrid = (num_pairs + threadsperblock - 1) // threadsperblock

    compute_density_kernel[blockspergrid, threadsperblock, stream](
        d_pairs, d_positions, d_masses, d_densities, params
    )

    compute_all_pair_forces[blockspergrid, threadsperblock, stream](
        d_pairs, d_velocities, d_positions, d_densities, d_masses, d_accels,
        np.float32(time), params
    )

    d_accels.copy_to_host(accels, stream=stream)
    stream.synchronize()

    for i in range(n):
        x, y = positions[i]
        dx = container_size - abs(x)
        dy = container_size - abs(y)
        if dx < 10:
            direction = np.array([np.sign(x),0], dtype=np.float32)
            F = 20
            accels[i] -= F * direction / masses[i]
        if dy < 10:
            direction = np.array([0,np.sign(y)], dtype=np.float32)
            F = 20
            accels[i] -= F * direction / masses[i]

    return accels



def rk4(time, bodies, args, dt):
    positions = np.array([body.position for body in bodies])
    masses = np.array([body.mass for body in bodies])
    velocities = np.array([body.velocity for body in bodies])

    k1_vel = compute_accels(time, positions, velocities, masses, args)
    k1_pos = velocities

    k2_vel = compute_accels(time + 0.5 * dt, positions + 0.5 * dt * k1_pos,
                            velocities + 0.5 * dt * k1_vel, masses, args)
    k2_pos = velocities + 0.5 * dt * k1_vel

    k3_vel = compute_accels(time + 0.5 * dt, positions + 0.5 * dt * k2_pos,
                            velocities + 0.5 * dt * k1_vel, masses, args)
    k3_pos = velocities + 0.5 * dt * k2_vel

    k4_vel = compute_accels(time + dt, positions + dt * k3_pos,
                            velocities + dt * k1_vel, masses, args)
    k4_pos = velocities + dt * k3_vel

    dpos = dt * (1/6) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
    dvel = dt * (1/6) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)

    return dpos, dvel

    
def reflect_boundaries(bodies, container_size, damp=0.1):
    for body in bodies:
        x, y = body.position
        vx, vy = body.velocity

        if abs(x) >= container_size:
            x = np.sign(x) * (container_size - 0.1)
            vx = -damp * vx
        if abs(y) >= container_size:
            y = np.sign(y) * (container_size - 0.1)
            vy = -damp * vy

        body.position = np.array([x, y])
        body.velocity = np.array([vx, vy])

def run_simulation(bodies, t_start, t_finish, n, args, memfrac=0.5):
    container_size = args['container_size']
    
    positions = np.array([body.position for body in bodies])
    masses = np.array([body.mass for body in bodies])
    velocities = np.array([body.velocity for body in bodies])
    
    # Initial conditions
    current_pos = positions
    current_vel = velocities

    pos = [current_pos.copy()]
    vel = [current_vel.copy()]
    filtered_t_vals = []
    
    # identify system memory and the size of the full arrays
    element_size = 8 # standard numpy float64
    full_array_size = 2 * element_size * n * len(bodies) # approximate array size in bytes
    mem = psutil.virtual_memory()
    ith = 1
    
    # If the full arrays are a significant portion of the full system memory limit the number of values saved
    if full_array_size > memfrac * mem.available:
        ith = int(full_array_size / (memfrac * mem.available)) # save every ith value
        print(f'Memory limit reached — reducing saved time steps to every {ith}th step.')
    
    dt = (t_finish - t_start) / (n - 1)
    t_vals = np.linspace(t_start,t_finish,n)
    i = 0
    for t in tqdm.tqdm(t_vals, desc="Running simulation"):
        # Extract current state from bodies (true current state)
        reflect_boundaries(bodies, container_size)
        current_pos = np.array([body.position for body in bodies])
        current_vel = np.array([body.velocity for body in bodies])
        
        # rk4 step
        dpos, dvel = rk4(t, bodies, args, dt)
        new_pos = current_pos + dpos
        new_vel = current_vel + dvel
        
        # save every ith value
        if i % ith == 0:
            pos.append(new_pos)
            vel.append(new_vel)
            filtered_t_vals.append(t)

        # update body positions and velocities
        for j, body in enumerate(bodies):
            body.position = new_pos[j]
            body.velocity = new_vel[j]
        i += 1
    return pos, vel, filtered_t_vals

# Simulation parameters
t_start = 0
t_finish = 50
n = 10000
container_size = 100
args = {
    'h': 1.0,
    'k': 10.0,
    'mu': 0.1,
    'rho_0': 1000.0,
    'container_size': container_size
}



bodies = []
# generate bodies
for _ in range(200):
    start_pos = np.array([np.random.uniform(-container_size, container_size), np.random.uniform(-container_size, container_size)])
    start_vel = np.random.uniform(-5, 5, 2)
    mass = 1
    bodies.append(Body(position=start_pos, velocity=start_vel, mass=mass))
    
    
# Run the simulation
start = time.perf_counter()
positions, velocities, times = run_simulation(bodies, t_start, t_finish, n, args)
end = time.perf_counter()
print(f'Simulation took {end - start:.2f} seconds.')

#find the output timestep
dt = abs(times[0] - times[1])

# Plot final positions
positions = np.array(positions)

# Create a figure and axis
fig, ax = plt.subplots()
"""
ax.set_xlim(np.min(positions[:,:,0]) - 0.5, np.max(positions[:,:,0]) + 0.5)
ax.set_ylim(np.min(positions[:,:,1]) - 0.5, np.max(positions[:,:,1]) + 0.5)
"""
ax.set_xlim(-container_size, container_size)
ax.set_ylim(-container_size, container_size)
ax.set_title("FLuid Simulation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)



# Create a text box for time display
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# One global scatter, instead of many
scatter = ax.plot([], [], 'o')[0]

def init():
    scatter.set_data([], [])
    time_text.set_text('')
    return [scatter, time_text]

def update(frame):
    x = positions[frame, :, 0]
    y = positions[frame, :, 1]
    scatter.set_data(x, y)
    time_text.set_text(f'Time: {frame * dt:.2f}')
    return [scatter, time_text]


# Create the animation
width, height = 1920, 1080  # example resolution
channels = 3  # RGB
bytes_per_pixel = 1  # 8-bit = 1 byte

frame_size_bytes = width * height * channels * bytes_per_pixel

total_size = frame_size_bytes * len(positions)

mem = psutil.virtual_memory()
memfrac = 0.2
ith = 1
if total_size > memfrac * mem.available:
    ith = int(total_size / (memfrac * mem.available)) # save every ith value


target_fps = 30  # frames per second
sim_seconds_per_frame = dt
frames_per_second_sim = 1 / sim_seconds_per_frame

frame_skip = max(1, int(frames_per_second_sim / target_fps))

if ith > frame_skip:
    frame_skip = ith
    print(f"Skipping every {frame_skip} frame(s) to match system memory constraints.")
else:
    print(f"Skipping every {frame_skip} frame(s) to match ~{target_fps} FPS in real time.")
ani = animation.FuncAnimation(fig, update, frames=range(0, len(positions), frame_skip), init_func=init,
                              blit=True, interval=1, repeat=True)

fps_out = 1 / (dt * frame_skip)
print(f"Saving video with fps={fps_out:.2f}")

ani.save("./animations/output.mp4", writer='ffmpeg', fps=fps_out)
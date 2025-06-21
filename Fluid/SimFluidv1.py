import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree
import psutil
import time
import tqdm

# === Data class for particles ===
class Body:
    def __init__(self, position, velocity, mass):
        self.position = position
        self.velocity = velocity
        self.mass = mass

# === SPH acceleration calculation ===
def compute_accels(time, positions, velocities, masses, args):
    gravity = np.array([0, -10])  # Constant gravity
    mu, gamma, h, box_limit = args
    eps2 = 1e-3
    r_cut = h
    n = len(positions)
    accels = np.zeros_like(positions)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=r_cut)

    density = np.zeros(n)
    accels += gravity
    
    for i in range(n):
        density[i] += masses[i] * 315 * (h**2)**3 / (64 * h**9 * np.pi)  # W(0)

    
    for i, j in pairs:
        r_vec = (positions[j] - positions[i])
        r2 = np.dot(r_vec, r_vec) + eps2  # softening added here
        r = np.sqrt(r2)
        W = 315 * (h**2 - r**2)**3 / (64 * h**9 * np.pi)
        density[i] += masses[j] * W
        density[j] += masses[i] * W

    for i, j in pairs:
        r_vec = positions[j] - positions[i]
        r = np.linalg.norm(r_vec)
        direction = r_vec / r
        delW = -r * 945 * (h**2 - r**2)**2 / (32 * h**9 * np.pi)
        lapW = -r * 945 * (h**2 - r**2) * (r**2 - 0.75 * (h**2 - r**2)) / (8 * h**9 * np.pi)

        p_i = iso * density[i]**gamma
        p_j = iso * density[j]**gamma

        accels[i] -= direction * delW * masses[j] * (p_j + p_i) / (2 * density[j])
        accels[j] += direction * delW * masses[i] * (p_i + p_j) / (2 * density[i])

        accels[i] += lapW * mu * masses[j] * (velocities[j] - velocities[i]) / density[j]
        accels[j] += lapW * mu * masses[i] * (velocities[i] - velocities[j]) / density[i]

    for i in range(n):
        x, y = positions[i]
        dx = box_limit - abs(x)
        dy = box_limit - abs(y)

        if dx < h:
            direction = np.array([np.sign(x), 0])
            accels[i] -= 1 * direction / masses[i]
        if dy < h:
            direction = np.array([0, np.sign(y)])
            accels[i] -= 10 * direction / masses[i]
        max_accel = 1e3
        norm = np.linalg.norm(accels[i])
        if norm > max_accel:
            accels[i] = accels[i] * (max_accel / norm)

    if np.any(np.isnan(accels)) or np.any(np.isinf(accels)):
        print("Warning: Invalid acceleration at time", time)
    return accels

# === RK4 integrator ===
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
                            velocities + 0.5 * dt * k2_vel, masses, args)
    k3_pos = velocities + 0.5 * dt * k2_vel

    k4_vel = compute_accels(time + dt, positions + dt * k3_pos,
                            velocities + dt * k3_vel, masses, args)
    k4_pos = velocities + dt * k3_vel

    dpos = dt * (1/6) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
    dvel = dt * (1/6) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)

    return dpos, dvel

# === Reflective boundary ===
def reflect_boundaries(bodies, boundary=10, damp=0.1):
    for body in bodies:
        x, y = body.position
        vx, vy = body.velocity

        if abs(x) >= boundary:
            x = np.sign(x) * (boundary - 0.1)
            vx = -damp * vx
        if abs(y) >= boundary:
            y = np.sign(y) * (boundary - 0.1)
            vy = -damp * vy

        body.position = np.array([x, y])
        body.velocity = np.array([vx, vy])

# === Main simulation runner ===
def run_simulation(bodies, t_start, t_finish, n, args, memfrac=0.5):
    positions = np.array([body.position for body in bodies])
    masses = np.array([body.mass for body in bodies])
    velocities = np.array([body.velocity for body in bodies])
    mu, gamma, h, box_limit = args
    
    current_pos = positions
    current_vel = velocities

    pos = [current_pos.copy()]
    vel = [current_vel.copy()]
    filtered_t_vals = []

    element_size = 8
    full_array_size = 2 * element_size * n * len(bodies)
    mem = psutil.virtual_memory()
    ith = 1
    if full_array_size > memfrac * mem.available:
        ith = int(full_array_size / (memfrac * mem.available))
        print(f'Memory limit reached — reducing saved time steps to every {ith}th step.')

    dt = (t_finish - t_start) / (n - 1)
    t_vals = np.linspace(t_start, t_finish, n)
    i = 0
    for t in tqdm.tqdm(t_vals, desc="Running simulation"):
        reflect_boundaries(bodies,box_limit)
        current_pos = np.array([body.position for body in bodies])
        current_vel = np.array([body.velocity for body in bodies])
        
        
        dpos, dvel = rk4(t, bodies, args, dt)
        new_pos = current_pos + dpos
        new_vel = current_vel + dvel

        if i % ith == 0:
            pos.append(new_pos)
            vel.append(new_vel)
            filtered_t_vals.append(t)

        for j, body in enumerate(bodies):
            body.position = new_pos[j]
            body.velocity = new_vel[j]
        i += 1
    return pos, vel, filtered_t_vals

# === Simulation constants ===
mu = 0.1           # Viscosity coefficient
h = 5.             # Smoothing length (also used as r_cut)
iso = 1e-1         # Isothermal compressibility factor
gamma = 7
box_limit = 10    # Bounding box limit
args = (mu, gamma, h, box_limit)     # Parameters for force computation

# === Initialization ===
t_start = 0
t_finish = 50
n = 10000

bodies = []
for _ in range(20):
    start_pos = np.array([np.random.uniform(-box_limit, box_limit), np.random.uniform(-box_limit, 0)])
    start_vel = np.random.uniform(-1, 1, 2)
    mass = 1
    bodies.append(Body(position=start_pos, velocity=start_vel, mass=mass))

start = time.perf_counter()
positions, velocities, times = run_simulation(bodies, t_start, t_finish, n, args)
finish = time.perf_counter()
print(f'Simulation completed in {finish - start:.2f}s')

dt = abs(times[1] - times[0])
positions = np.array(positions)

# === Animation ===
fig, ax = plt.subplots()
ax.set_xlim(-box_limit, box_limit)
ax.set_ylim(-box_limit, box_limit)
ax.set_title("SPH Simulation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
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

width, height = 1920, 1080
channels = 3
bytes_per_pixel = 1
frame_size_bytes = width * height * channels * bytes_per_pixel
total_size = frame_size_bytes * len(positions)
mem = psutil.virtual_memory()
memfrac = 0.8
ith = 1
if total_size > memfrac * mem.available:
    ith = int(total_size / (memfrac * mem.available))

target_fps = 30
sim_seconds_per_frame = dt
frames_per_second_sim = 1 / sim_seconds_per_frame
frame_skip = max(1, int(frames_per_second_sim / target_fps))
if ith > frame_skip:
    frame_skip = ith
    print(f"Skipping every {frame_skip} frame(s) to match system memory constraints.")
else:
    print(f"Skipping every {frame_skip} frame(s) to match ~{target_fps} FPS in real time.")

ani = animation.FuncAnimation(fig, update, frames=range(0, len(positions), frame_skip),
                              init_func=init, blit=True, interval=1, repeat=True)

fps_out = 1 / (dt * frame_skip)
print(f"Saving video with fps={fps_out:.2f}")
ani.save("./animations/output.mp4", writer='ffmpeg', fps=fps_out)


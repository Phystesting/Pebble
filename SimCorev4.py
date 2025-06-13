import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from functools import partial
import psutil
import time


# fluid in a box under gravity
#verlet works if the force isn't velocity dependent

# stores the current position velocity and mass of each particle
class Body:
    def __init__(self, position, velocity, mass):
        self.position = position
        self.velocity = velocity
        self.mass = mass

# leonard jones force 4 * epsilon * (6*(sigma/x) ** 6 - 12*(sigma/x) ** 12) / x 
def compute_accels(positions, masses, args):
    epsilon, sigma = args
    r_cut = 2.5 * sigma
    n = len(positions)
    accels = np.zeros([n, 2])
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=r_cut)
    # forces on every particle
    gravity = np.array([0, -10])  # constant downward force
    for k in range(n):
        accels[k] += gravity
    # forces between pairs
    for i, j in pairs:
        r_vec = positions[j] - positions[i]
        r = np.linalg.norm(r_vec)
        direction = r_vec / r
        # Safe direction — use original r_vec but normalize with original r
        if r > 1:  # avoid division by zero
            F = 4 * epsilon * (6 * (sigma / r)**6 - 12 * (sigma / r)**12) / r
        else:
            F = 4 * epsilon * (6 * (sigma)**6 - 12 * (sigma)**12)

        # Apply force and gravity
        accels[i] += (F * direction) / masses[i]
        accels[j] -= (F * direction) / masses[j]

    return accels


def rk4(bodies, args, dt):
    #unpack bodies
    positions = np.array([body.position for body in bodies])
    masses = np.array([body.mass for body in bodies])
    velocities = np.array([body.velocity for body in bodies])
    
    #rk4 steps
    k1_vel = compute_accels(positions, masses,args)
    k1_pos  = velocities
    
    k2_vel = compute_accels(positions + 0.5 * dt * k1_pos,masses,args)
    k2_pos = velocities + 0.5 * dt * k1_vel
    
    k3_vel = compute_accels(positions + 0.5 * dt * k2_pos,masses,args)
    k3_pos = velocities + 0.5 * dt * k2_vel
    
    k4_vel = compute_accels(positions + 0.5 * dt * k3_pos,masses,args)
    k4_pos = velocities + 0.5 * dt * k3_vel
    
    #calculate change in velocity and position for each body
    dpos = dt * (1/6) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
    dvel = dt * (1/6) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
    
    return dpos, dvel

def verlet(bodies, args, dt):
    
    #unpack bodies
    positions = np.array([body.position for body in bodies])
    masses = np.array([body.mass for body in bodies])
    velocities = np.array([body.velocity for body in bodies])
    
    f1 = compute_accels(positions, masses,args)
    dpos = dt * velocities + 0.5 * (dt**2) * f1
    f2 = compute_accels(positions + dpos, masses,args)
    dvel = 0.5 * dt * (f2 + f1)
    return dpos, dvel
    
def boundaries(bodies):
    positions = np.array([body.position for body in bodies])
    velocities = np.array([body.velocity for body in bodies])
    boundary = 100
    for i in range(len(bodies)):
        x, y = positions[i]
        vx, vy = velocities[i]

        # Reflect off top/bottom
        if abs(y) >= boundary:
            vy = -0.1 * vy
            #print(y)
            y = np.sign(y) * (boundary - 0.1)  # push slightly in
            #print(y)

        # Reflect off left/right
        if abs(x) >= boundary:
            vx = -0.1 * vx
            #print(x)
            x = np.sign(x) * (boundary - 0.1)  # push slightly in
            #print(x)

        positions[i] = [x, y]
        velocities[i] = [vx, vy]

    # Update the bodies
    for i, body in enumerate(bodies):
        body.position = positions[i]
        body.velocity = velocities[i]


def run_simulation(bodies, t_start, t_finish, n, memfrac=0.5):
    positions = np.array([body.position for body in bodies])
    masses = np.array([body.mass for body in bodies])
    velocities = np.array([body.velocity for body in bodies])
    
    # Initial conditions
    current_pos = positions
    current_vel = velocities

    pos = [current_pos.copy()]
    vel = [current_vel.copy()]
    filtered_t_vals = []
    
    #identify system memory and the size of the full arrays
    element_size = 8 # standard numpy float
    full_array_size = 2 * element_size * n * len(bodies) # approximate array size in bytes
    mem = psutil.virtual_memory()
    ith = 1
    
    #If the full arrays are a significant portion of the full system memory limit the number of values saved
    if full_array_size > memfrac * mem.available:
        ith = int(full_array_size / (memfrac * mem.available)) # save every ith value
        print(f'Memory limit reached — reducing saved time steps to every {ith}th step.')
    dt = (t_finish - t_start) / (n - 1)
    t_vals = np.linspace(t_start,t_finish,n)
    i = 0
    for t in t_vals:
        # Extract current state from bodies (true current state)
        boundaries(bodies)
        current_pos = np.array([body.position for body in bodies])
        current_vel = np.array([body.velocity for body in bodies])
        
        # RK4 step
        dpos, dvel = verlet(bodies, args, dt)
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
epsilon = 1.0
sigma = 1.0
args = (epsilon, sigma)


bodies = []
# generate bodies
for _ in range(100):
    start_pos = np.random.uniform(-100,0,2)
    start_vel = np.random.uniform(-5,5,2)
    mass = 1
    bodies.append(Body(position=start_pos, velocity=start_vel, mass=mass))


# Run the simulation
start = time.perf_counter()
positions, velocities, times = run_simulation(bodies, t_start, t_finish, n)
finish = time.perf_counter()
print(f'Simulation completed in {finish-start}s')

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
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_title("Lennard-Jones Simulation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Create scatter plot for each body
scatters = [ax.plot([], [], 'o', label=f'Body {i+1}')[0] for i in range(len(bodies))]
#ax.legend()

# Create a text box for time display
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Initialization function
def init():
    for scatter in scatters:
        scatter.set_data([], [])
    time_text.set_text('')
    return scatters + [time_text]

# Animation update function
def update(frame):
    for i, scatter in enumerate(scatters):
        scatter.set_data(positions[frame, i, 0], positions[frame, i, 1])
    current_time = frame * dt
    time_text.set_text(f'Time: {current_time:.2f}')
    return scatters + [time_text]

# Create the animation
target_fps = 30  # frames per second
sim_seconds_per_frame = dt
frames_per_second_sim = 1 / sim_seconds_per_frame

frame_skip = max(1, int(frames_per_second_sim / target_fps))
print(f"Skipping every {frame_skip} frame(s) to match ~{target_fps} FPS in real time.")
ani = animation.FuncAnimation(fig, update, frames=range(0, len(positions), frame_skip), init_func=init,
                              blit=True, interval=1, repeat=True)

plt.show()


        
        
        

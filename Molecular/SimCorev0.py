import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from functools import partial

#most basic N_body solver

# stores the current position velocity and mass of each particle
class Body:
    def __init__(self, position, velocity, mass):
        self.position = position
        self.velocity = velocity
        self.mass = mass

# leonard jones force 4 * epsilon * (6*(sigma/x) ** 6 - 12*(sigma/x) ** 12) / x 
def compute_accels(positions,masses,args):
    epsilon, sigma = args
    n = len(positions[0])
    accels = np.zeros([n, 2])
    #first loop selects body being acted on
    for i in range(n):
        #second loop selects interacting body
        for j in range(i + 1,n):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            # compute force
            F = 4 * epsilon * (6*(sigma/r) ** 6 - 12*(sigma/r) ** 12) / r
            accels[i] += (F * r_vec / r) / masses[i]
            accels[j] -= (F * r_vec / r) / masses[j]
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
    
def run_simulation(bodies, t_start, t_finish, n):
    positions = np.array([body.position for body in bodies])
    masses = np.array([body.mass for body in bodies])
    velocities = np.array([body.velocity for body in bodies])
    pos = [positions]
    vel = [velocities]
    dt = (t_finish - t_start) / (n - 1)
    t_vals = np.linspace(t_start,t_finish,n)
    for t in t_vals:
        dpos, dvel = rk4(bodies, args, dt)

        new_pos = pos[-1] + dpos
        new_vel = vel[-1] + dvel

        pos.append(new_pos)
        vel.append(new_vel)

        # update body positions and velocities
        for i, body in enumerate(bodies):
            body.position = new_pos[i]
            body.velocity = new_vel[i]
    return pos, vel, t_vals
        
# Simulation parameters
t_start = 0
t_finish = 30
n = 10000
dt = (t_finish - t_start) / (n - 1)
epsilon = 1.0
sigma = 1.0
args = (epsilon, sigma)

# Create two bodies in a simple orbit-like setup
body1 = Body(position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), mass=1.0)
body2 = Body(position=np.array([1.2, 0.0]), velocity=np.array([0.0, 0.1]), mass=1.0)

bodies = [body1, body2]

# Run the simulation
positions, velocities, times = run_simulation(bodies, t_start, t_finish, n)

# Plot final positions
positions = np.array(positions)
# Create a figure and axis
fig, ax = plt.subplots()
ax.set_xlim(np.min(positions[:,:,0]) - 0.5, np.max(positions[:,:,0]) + 0.5)
ax.set_ylim(np.min(positions[:,:,1]) - 0.5, np.max(positions[:,:,1]) + 0.5)
ax.set_title("Two-Body Lennard-Jones Simulation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Create scatter plot for each body
scatters = [ax.plot([], [], 'o', label=f'Body {i+1}')[0] for i in range(len(bodies))]
ax.legend()

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
ani = animation.FuncAnimation(fig, update, frames=len(positions), init_func=init,
                              blit=True, interval=1, repeat=False)

plt.show()


        
        
        

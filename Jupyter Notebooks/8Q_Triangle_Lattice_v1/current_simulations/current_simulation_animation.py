import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from current_simulation import CurrentSimulations, generate_basis

num_qubits = 4
num_particles = 2

num_states = math.comb(num_qubits, num_particles)

J_parallel = 1
J_perp = J_parallel

detuning = [1000]*num_qubits
detuning[0] = 0
detuning[1] = 0


simulation = CurrentSimulations(num_qubits, num_particles, J_parallel, J_perp, detuning)

psi0 = simulation.get_resonant_ground_state()
print(psi0)
times = np.linspace(0, 1, 101)

result = simulation.run_simulation(psi0, times, resonant=False)

# --- Parameters ---
num_steps = len(times)  # Total number of timesteps
basis = generate_basis(num_qubits, num_particles)
dim = len(basis)
print("Basis states (index: state):")
for idx, state in enumerate(basis):
    print(idx, state)

state_vectors = result.states

# --- Qubit positions ---
positions = {
    0: (0, 0),
    1: (1, 0),
    2: (0, 1),
    3: (1, 1)
}

# --- Visualization Functions ---
def compute_qubit_contributions(state_vector, basis):
    contributions = {i: 0 + 0j for i in range(num_qubits)}
    for amp, state in zip(state_vector, basis):
        for qubit, occ in enumerate(state):
            if occ == 1:
                contributions[qubit] += amp
    return contributions

def phase_to_color(z, cmap=plt.cm.hsv):
    phase = np.angle(z)
    hue = (phase + np.pi) / (2 * np.pi)
    brightness = min(abs(z), 1.0)
    base_color = np.array(cmap(hue))
    color = brightness * base_color[:3]
    return np.clip(color, 0, 1)

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Create a circle patch for each qubit.
circles = {}
radius = 0.15
for qubit, (x, y) in positions.items():
    circle = plt.Circle((x, y), radius, color='white', ec='black', lw=2)
    ax.add_patch(circle)
    circles[qubit] = circle

title = ax.text(0.5, 1.45, '', ha='center', transform=ax.transAxes, fontsize=12)

# --- Update Function ---
def update_plot(time_idx):
    title.set_text(f"Time step: {time_idx}")
    state_vector = state_vectors[time_idx]
    contributions = compute_qubit_contributions(state_vector, basis)
    
    for qubit, circle in circles.items():
        color = phase_to_color(contributions[qubit])
        circle.set_facecolor(color)

# --- Animation Setup ---
def animate(time_idx):
    update_plot(time_idx)
    slider.set_val(time_idx)  # Update the slider value
    return circles.values()

ani = FuncAnimation(fig, animate, frames=num_steps, interval=100, blit=False)

# --- Interactive Slider ---
ax_slider = plt.axes([0.2, 0.02, 0.4, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Time', 0, num_steps - 1, valinit=0, valstep=1)

# Update the plot when the slider is moved
def on_slider_change(val):
    time_idx = int(slider.val)
    update_plot(time_idx)
    fig.canvas.draw_idle()

slider.on_changed(on_slider_change)

# --- Animation Control ---
is_animating = [True]  # Use a mutable object to allow modification inside functions

def start_animation(event):
    is_animating[0] = True
    ani.event_source.start()  # Start the animation

def stop_animation(event):
    is_animating[0] = False
    ani.event_source.stop()  # Stop the animation

# --- Add Start/Stop Buttons ---
ax_start = plt.axes([0.8, 0.02, 0.1, 0.04])  # Position for the start button
ax_stop = plt.axes([0.7, 0.02, 0.1, 0.04])   # Position for the stop button

btn_start = Button(ax_start, 'Start')
btn_stop = Button(ax_stop, 'Stop')

btn_start.on_clicked(start_animation)
btn_stop.on_clicked(stop_animation)

# --- Show Plot ---
plt.show()
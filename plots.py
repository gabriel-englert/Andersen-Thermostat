import numpy as np
from modules import *
plt.style.use("ggplot")

data = load_simulation_data("data_temp_3.0_n_108_dens_0.8442_dt_0.001_nu_0.5")
nu = data["nu"]
v_list = data["velocities"]
total_steps = data["totalsteps"]
print(v_list.shape)
temp = data["initial_temp"]
steps = 40000
last_50_velocities = v_list[-steps:]  # Shape: (50, n, D)
speeds = np.linalg.norm(last_50_velocities, axis=2).flatten()  # Flatten to 1D array

# Define the theoretical Maxwell-Boltzmann distribution
def maxwell_boltzmann_pdf(v, temp, m=1.0):
    """Compute the Maxwell-Boltzmann PDF for a given speed."""
    factor = np.sqrt(2 / np.pi) * (m / temp)**(3/2)
    return factor * v**2 * np.exp(-m * v**2 / (2 * temp))

# Plotting
bins = 50  # Number of histogram bins
v_range = np.linspace(0, np.max(speeds), 1000)  # Range for the theoretical PDF

plt.hist(speeds, bins=bins, density=True, alpha=0.6, label="Velocidades da Simulação")
plt.plot(v_range, maxwell_boltzmann_pdf(v_range, temp), label="Distribuição de Maxwell-Boltzmann", color='red')
plt.xlabel("Velocidade")
plt.ylabel("Densidade de Probabilidade")
plt.title(f"Distribuição de Velocidades")
plt.legend()
plt.grid()
plt.savefig(f"plots/velocity_distribution_{temp}.png")
plt.show()
k_list = np.zeros(total_steps)
for i in range(0,total_steps):
    k_list[i] = k_energy(v_list[i])
mean = np.mean(k_list[-steps:])/3
sd = np.std(k_list[-steps:]/3)
print(mean)
print(sd)
t_list = range(0,total_steps)

x = np.linspace(0,total_steps,10000)
y = np.array([temp for i in x])
mean_y = np.array([mean for i in x])
plt.plot(t_list,k_list/3,label="temperatura simulada",lw=0.7)
plt.plot(x,y,label="temperatura do reservatório",linestyle="dashed",c="black")
plt.plot(x,mean_y,label=f"mean temperature last {steps} steps",linestyle="dotted",c="blue")
plt.title("Temperatura no Termostato de Andersen")
plt.xlabel("Número de Passos")
plt.ylabel("Temperatura")
plt.legend()
plt.ylim(0,4)
plt.savefig(f"plots/temp_{temp}_totalsteps_{total_steps}_nu_{nu}.png")
plt.show()
print(k_list/3)
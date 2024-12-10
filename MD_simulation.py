import numpy as np
from modules import *
#parameters units
density = 0.8442
D = 3 #dimension of the simulation
dt = 0.001 #time interval for integration
T = 6 #total time of simulation
n = 108 #number of particles
L = int(np.cbrt(n/density)) #side of the box

sig = 1.0
eps = 1.0
m = 1.0
#parameters in dimensionless units
L = L/sig
dt = dt*np.sqrt(eps/(m*sig**2))
T = T*np.sqrt(eps/(m*sig**2))
temp = 2.0

total_steps = int(T/dt)
kb = 1.0
rc = 2.5*sig #critic r for truncated potential


#array of positions
#r = np.random.rand(n,D)*L #initialize random positions
r = initialize_lattice(n,D,L)
r = r/sig
r_list = np.zeros((total_steps,n,D)) #array to store time evolution of positions
r_list[0] = r

#v = (np.random.rand(n,D)-0.5) #initialize velocities randomly between -0.5 and 0.5
#v = maxwell_boltzmann_velocities(n,D,temp,m)
v = random_v(n,D,temp)
plt.hist(v,bins=10)
plt.show()
v_list = np.zeros((total_steps,n,D))

k_list = np.zeros(total_steps)
u_list = np.zeros(total_steps)
temp_list = np.zeros(total_steps)
#dynamics
for t in range(total_steps):
    print(t)
    r, v = velocity_verlet(r,v,dt,L,sig,eps,n,D)
    r_list[t] = r
    v_list[t] = v
    k = k_energy(v)
    
    #u = np.array([LJpot_truncated_shifted(r,i,sig,eps,rc) for i in range(n)])
    u = np.array([LJpot_truncated_shifted(r,i,sig,eps) for i in range(n)])
    u = np.sum(u)
    T_t = k/D
    temp_list[t] = T_t
    k_list[t] = k
    u_list[t] = u/(n)
    dump(r,t,L,D)
print(np.mean(temp_list[-1000:]))
print(np.std(temp_list[-1000:]))
plt.hist(temp_list,bins=10)
plt.savefig("temperatures.png")
plt.show()
plt.plot(range(total_steps),k_list)
plt.plot(range(total_steps),u_list)
plt.plot(range(total_steps),u_list + k_list)
plt.savefig("energy_basic.png")

#plt.plot(range(total_steps),u_list+k_list)
plt.show()


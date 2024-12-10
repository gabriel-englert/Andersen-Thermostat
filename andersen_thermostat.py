import numpy as np
from modules import *

'''independent Parameters'''
density = 0.8442
D = 3 #dimension of the simulation
dt = 0.001 #time interval for integration
T = 0.5 #total time of simulation
n = 108 #number of particles
sig = 1.0 #parameter of the potential
eps = 1.0 #parameter of the potential
m = 1.0 #mass of particles
temp = 0.1
kb = 1.0
rc = 2.5 #critic r for truncated potential
nu = 0.05 #collision rate with heat bath

'''Dependent Parameters'''
L = int(np.cbrt(n/density)) #side of the box, dependent on n and density
total_steps = int(T/dt) #total steps of simulation

'''Initialization of Positions'''
r = initialize_lattice(n,D,L) #array of positions
r_list = np.zeros((total_steps,n,D)) #array to store time evolution of positions
r_list[0] = r

'''Initialization of Velocities'''
#v = maxwell_boltzmann_velocities(n,D,temp,m)
v = random_v(n,D,temp) #function from modules.py
v_list = np.zeros((total_steps,n,D)) #array to store time evolution of velocities

'''Potential Energy array'''
u_list = np.zeros(total_steps)

'''Standard Deviation of the gaussian for velocity sampling'''
std_dev = np.sqrt(temp)


'''Dynamics'''
for t in range(total_steps):
    if t%1000==0: #to track the progress on terminal
        print(t)
    r, v = velocity_verlet(r,v,dt,L,sig,eps,n,D) #integration function returns next steps' positions and velocities
    #the difference to andersen is just this next for loop    
    for i in range(n):
        p = np.random.rand() #ramdom number
        j = np.random.randint(0,n) #random particle
        if p<nu*dt:
            v[j] = np.random.normal(loc=0.0,scale=std_dev,size=D) #sample from gaussian of given temperature
    
    r_list[t] = r
    v_list[t] = v
    u = np.sum(np.array([LJpot_truncated_shifted(r,i,sig,eps) for i in range(n)])) #tptal potential energy
    u_list[t] = u/n
    
save_simulation_data(u_list,v_list,total_steps,dt,temp,density,n,nu)






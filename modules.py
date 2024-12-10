import numpy as np
import matplotlib.pyplot as plt

def initialize_lattice(n, D, L):
    '''
    Initialize n particles in a D-dimensional lattice inside a box of size L.
    
    Parameters:
    n -- Number of particles
    D -- Number of dimensions
    L -- Side length of the simulation box
    
    Returns:
    positions -- (n, D) array of particle positions in the lattice
    '''
    
    particles_per_dim = int(np.ceil(n ** (1 / D))) # Determine the number of particles per dimension

   
    linspace = np.linspace(0, L, particles_per_dim, endpoint=False)  # Generate the lattice points in each dimension
    grid = np.meshgrid(*[linspace] * D, indexing="ij")

    
    positions = np.vstack([g.ravel() for g in grid]).T # Stack the grid points into a list of coordinates

    
    positions = positions[:n] # Trim to the required number of particles

    return positions


def apply_periodic_boundary(r,L,D):
    '''
    Initialize n particles in a D-dimensional lattice inside a box of size L.
    
    Parameters:
    r -- Array of positions of particles in a given time
    L -- Side length of the simulation box
    D -- Number of dimensions
    
    Returns:
    r -- (n, D) Array of updated positions according to boundary conditions
    '''
    for i in range(r.shape[0]):
        for j in range(D):
            if r[i][j] > L:
                r[i][j] %= L
            if r[i][j] < 0:
                r[i][j] %= L
    return r

def LJpot(r,i,sig=1.0,eps=1.0):
    drv = r-r[i] #distance in each dimension
    drv = np.delete(drv,i,0) #remove ith element (remove itself)
    dr = [np.sqrt(a[0]**2+a[1]**2+a[2]**2) for a in drv] #absolute distance
    r6 = (1/np.array(dr))**6
    r12 = (1/np.array(dr))**12
    LJP = 4.0 * np.sum(r12-r6)
    return LJP

def LJ(x,sigma=1):
    r6 = (1/x)**6
    r12 = (1/x)**12
    LJP = 4.0*(r12-r6)
    return LJP


def LJpot_truncated_shifted(r, i, sig=1.0, eps=1.0, rc=2.5):
    """
    Compute the truncated and shifted Lennard-Jones potential for the i-th particle.

    Parameters:
    r -- Array of particle positions, shape (n, D)
    i -- Index of the particle
    sig -- Sigma parameter of the Lennard-Jones potential
    eps -- Epsilon parameter of the Lennard-Jones potential
    rc -- Cutoff distance

    Returns:
    LJP -- Truncated and shifted Lennard-Jones potential for the i-th particle
    """
    # Distance vectors from particle i
    drv = r - r[i]
    drv = np.delete(drv, i, axis=0)  # Remove the i-th particle
    dr = np.linalg.norm(drv, axis=1)  # Magnitude of distance vectors
    
    # Compute the Lennard-Jones potential at r_c
    rc6 = (1.0 / rc) ** 6
    rc12 = (1.0/rc)**12
    u_rc = 4.0 * (rc12 - rc6)  # Lennard-Jones potential at r_c

    # Filter distances within the cutoff radius
    within_cutoff = dr < rc
    dr = dr[within_cutoff]

    # Compute the Lennard-Jones potential for valid distances
    r6 = (1.0 / dr) ** 6
    r12 = (1.0/dr)**12
    u_r = 4.0 * (r12 - r6)

    # Apply truncation and shifting
    u_tr_sh = u_r - u_rc

    # Return the total truncated and shifted potential
    LJP = np.sum(u_tr_sh)

    return LJP

def dLJp(r,i,sig,eps,D,L):
    drv = periodic_drv(r,i,L)
    dr = periodic_d(r,i,L)
    r8 = (1.0/dr)**7
    r14 = 2.0*(1.0/dr)**13
    r814 = r14-r8
    r814v =np.transpose(np.transpose(drv)*r814)
    r814vs =np.sum(r814v,axis=0)
    dLJP=24.0*(r814vs)
    return dLJP
    
def dLJp_truncated(r, i,sig,eps,D,L,rc):
    """
    Compute the truncated, shifted, and dimensionless Lennard-Jones force.

    Parameters:
    r  -- Array of particle positions, shape (n, D), in dimensionless units.
    i  -- Index of the particle for which to compute the force.
    rc -- Dimensionless cutoff distance (default: 2.5).
    L  -- Dimensionless box size (default: 100).

    Returns:
    dLJP -- Dimensionless Lennard-Jones force vector for the i-th particle.
    """
    # Compute periodic distances
    drv = periodic_drv(r, i, L)  # Shape: (n-1, D)
    dr = periodic_d(r, i, L)    # Shape: (n-1,)


    # Compute force components
    r8 = (1.0 / dr) ** 8
    r14 = 2.0 * (1.0 / dr) ** 14
    r814 = r14 - r8
        # Apply cutoff: Set contributions outside rc to zero
    r814[dr >= rc] = 0  # Zero out scaling factor beyond cutoff
    drv[dr >= rc] = 0  
    # Combine vector components
    r814v = np.transpose(np.transpose(drv) * r814)  # Shape: (n_cutoff, D)
    r814vs = np.sum(r814v, axis=0)  # Sum contributions (shape: D)

    # Dimensionless force
    dLJP = 24.0 * r814vs

    return dLJP

def velocity_verlet(r,v,dt,L,sig,eps,n,D,m=1,rc=2.5):
    F = -np.array([dLJp_truncated(r,i,sig,eps,D,L,rc) for i in range(n)])
    a = F/m
    newv = v + 0.5*a*dt
    newr = r + newv*dt
    newr = apply_periodic_boundary(newr,L,D)
    F = -np.array([dLJp_truncated(newr,i,sig,eps,D,L,rc) for i in range(n)])
    a = F/m
    v = newv + 0.5*a*dt
    r = apply_periodic_boundary(newr,L,D)
    return r,v

def dump(r,t,L,D):
    '''
    Saves files in the "dump" folder for visualization on the Ovito software
    '''
    fname="dump/t"+str(t)+".dump"
    f=open(fname,"w")
    f.write("ITEM: TIMESTEP\n")
    f.write(str(t)+"\n") #time step
    f.write("ITEM: NUMBER OF ATOMS\n")
    f.write(str(len(r))+"\n") # number of atoms
    f.write("ITEM: BOX BOUNDS pp pp pp\n") #pp = periodic BCs
    f.write("0 "+str(L)+"\n")
    f.write("0 "+str(L)+"\n")
    f.write("0 "+str(L)+"\n")
    f.write("ITEM: ATOMS id x y z\n")
    if D == 3:
        for i in range(len(r)):
            f.write(str(i)+" "+str(r[i][0])+" "+str(r[i][1])+" "+str(r[i][2])+"\n")
    if D == 2:
        for i in range(len(r)):
            f.write(str(i)+" "+str(r[i][0])+" "+str(r[i][1])+" "+str(0)+"\n")
    f.close


def periodic_d(r,i,L):
    drv = r - r[i]  # Shape: (n, D)
    
    # Apply periodic boundary conditions
    drv = drv - L * np.round(drv / L)
    
    # Remove the ith particle (distance to itself)
    drv = np.delete(drv, i, axis=0)
    
    # Compute magnitudes of the distance vectors
    distances = np.linalg.norm(drv, axis=1)  # Shape: (n-1,)
    return distances

def periodic_drv(r,i,L):
    drv = r - r[i]  # Shape: (n, D)
    
    # Apply periodic boundary conditions
    drv = drv - L * np.round(drv / L)
    
    # Remove the ith particle (distance to itself)
    drv = np.delete(drv, i, axis=0)
    return drv


def k_energy(v,m=1.0):
    k = np.mean(np.sum(m*v**2, axis=1))
    return k


def maxwell_boltzmann_velocities(n, D, temp, m=1.0, kb=1.0):
    '''
    Generate velocities for particles from the Maxwell-Boltzmann distribution.
    
    Parameters:
    n -- Number of particles
    D -- Number of dimensions
    T -- Temperature (in units where energy is dimensionless)
    m -- Particle mass (in units where mass is dimensionless)
    k_B -- Boltzmann constant (default is 1.0 for dimensionless units)
    
    Returns:
    velocities -- (n, D) array of velocities
    '''
    # Standard deviation of the Maxwell-Boltzmann distribution
    std_dev = np.sqrt(temp)
    
    # Generate velocities from a normal distribution
    velocities = np.random.normal(loc=0.0, scale=std_dev, size=(n, D))
    com_velocity = np.mean(velocities, axis=0)
    
    # Adjust velocities to make the Center of Mass(COM) velocity zero
    velocities -= com_velocity
    
    return velocities


def random_v(n,D,temp):
    '''
    Start with random velocities scaled to be in the given temperature
    COM has zero velocity
    '''
    v = (np.random.rand(n,D)-0.5)
    com_velocity = np.mean(v, axis=0)
    # Adjust velocities to make the COM velocity zero
    v -= com_velocity
    instant_temp = np.mean(np.sum(v**2, axis=1)) / D
    scaling_factor = np.sqrt(temp/instant_temp)
    v = scaling_factor*v
    return v


def save_simulation_data(potential_energy, velocities, totalsteps, dt, temp, density, n, nu):
    """
    Save simulation data and parameters to a .dat file.

    Parameters:
    - file_name (str): The name of the output .dat file.
    - potential_energy (numpy array): Array containing the potential energy for each time step.
    - velocities (numpy array): Array containing the velocities for each time step.
    - totalsteps (int): Total number of simulation steps.
    - dt (float): Time step used in the simulation.
    - final_temp (float): Final temperature after simulation.
    - initial_temp (float): Initial temperature at the start of simulation.
    - density (float): Density of the system.
    - n (int): Number of particles.
    - nu (float): Frequency for Andersen thermostat.
    """
    file_name = f"data/data_temp_{temp}_n_{n}_dens_{density}_dt_{dt}_nu_{nu}"
    # Open the file in write mode
    with open(file_name, 'w') as file:
        # Write the header with simulation parameters
        file.write(f"# Total Steps: {totalsteps}\n")
        file.write(f"# Time Step (dt): {dt}\n")
        file.write(f"# Initial Temperature: {temp}\n")
        file.write(f"# Density: {density}\n")
        file.write(f"# Number of Particles (n): {n}\n")
        file.write(f"# Andersen Thermostat Frequency (nu): {nu}\n")
        file.write("\n")  # Blank line separating header from data
        
        # Write column headers for the data
        file.write("# Time Step | Potential Energy | Velocities\n")
        
        # Write the data: time steps, potential energy, and velocities
        for t in range(totalsteps):
            file.write(f"{t} {potential_energy[t]} {velocities[t].tolist()}\n")

import ast
def load_simulation_data(file_name):
    """
    Load simulation data and parameters from a .dat file.

    Parameters:
    - file_name (str): The name of the .dat file to load.

    Returns:
    - A dictionary containing the parameters and the data arrays.
    """
    data = {
        'totalsteps': None,
        'dt': None,
        'initial_temp': None,
        'density': None,
        'n': None,
        'nu': None,
        'potential_energy': [],
        'velocities': []
    }

    with open(file_name, 'r') as file:
        lines = file.readlines()
        
        # Extract parameters from header lines
        for line in lines:
            if line.startswith("# Total Steps:"):
                data['totalsteps'] = int(line.split(":")[1].strip())
            elif line.startswith("# Time Step (dt):"):
                data['dt'] = float(line.split(":")[1].strip())
            elif line.startswith("# Initial Temperature:"):
                data['initial_temp'] = float(line.split(":")[1].strip())
            elif line.startswith("# Density:"):
                data['density'] = float(line.split(":")[1].strip())
            elif line.startswith("# Number of Particles (n):"):
                data['n'] = int(line.split(":")[1].strip())
            elif line.startswith("# Andersen Thermostat Frequency (nu):"):
                data['nu'] = float(line.split(":")[1].strip())

        # Extract data lines (after the header)
        for line in lines:
            if line.startswith("# Time Step |"):
                continue
                # Skip the header line
            if line.strip():  # Skip empty lines
                parts = line.split(maxsplit=2)
                try:
                    # Check if the line contains valid data
                    time_step = int(parts[0])  # First part is the time step (integer)
                    potential_energy = float(parts[1])  # Second part is the potential energy (float)
                    
                    # Parse the velocities (the third part, which is a list of floats)
                    velocities = ast.literal_eval(parts[2])
                    
                    # Append the values to the data
                    data['potential_energy'].append(potential_energy)
                    data['velocities'].append(velocities)
                except ValueError:
                    # If there is an error parsing the line, skip it
                    continue

    # Convert lists to numpy arrays
    data['potential_energy'] = np.array(data['potential_energy'])
    data['velocities'] = np.array(data['velocities'])

    return data

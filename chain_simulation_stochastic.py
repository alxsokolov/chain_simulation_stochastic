import numpy as np
import matplotlib.pyplot as plt
import logging 
import multiprocessing as mp
import os
import sys
import pdb
import random
from mpl_toolkits import mplot3d

logging.basicConfig(filename='chain_simulation_stochastic_displacements.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

#constants and parameters
results=[]

def ind(i):
    return np.where(n==i)

def initial_profile_vel(x):
    return np.heaviside(x+20, 1) - np.heaviside(x-20, 1)

def initial_profile_disp(x):
    return np.heaviside(x+20, 1) - np.heaviside(x-20, 1)

def run_simulations(N, Nrealisations, n_steps, dt, i_thread, dump_interval):
    logging.info('%d-th thread started simulation of %d realisations\n', i_thread, Nrealisations)
    ddu=np.zeros(N)
    du=np.zeros(N)
    u=np.zeros(N)
    du2_averaged=np.zeros(N)
    rho = np.random.RandomState(123862498*i_thread)
    r = np.random.RandomState(54984358*i_thread)
    #averaging loop
    for j in np.arange(0, Nrealisations):
        #initial conditions
        #stochastic velocities
        for i in np.arange(0, N):
            du[i]=rho.normal()*initial_profile_vel(n[i])
        #stochastic displacements
        #zeroth particle displacement is zero
        u[int((N-1)/2)]=0
        #recursive formula for particles to the right
        for i in np.arange(int((N-1)/2)+1, N):
            u[i]=u[i-1]-r.normal()*initial_profile_disp(n[i])
        #recursive formula for particles to the left
        for i in np.arange(int((N-1)/2)+1, 0, -1):
            u[i-1]=u[i]+r.normal()*initial_profile_disp(n[i-1])
        #du=np.multiply(np.random.normal(0, 1, N), initial_profile(n))
        #integration loop
        for step in np.arange(0, n_steps):
            #calculate forces
            #boundary conditions
            #free ends
            ddu[0]=u[1] - u[0]
            ddu[N-1]=u[N-2] - u[N-1]
            #periodic boundary
            #ddu[0]=u[1] - 2*u[0] + u[N-1]
            #ddu[N-1]=u[N-2] - 2*u[N-1] + u[0]
            #inner particles
            for i in np.arange(1, N-1):
                ddu[i]=u[i+1]-2*u[i]+u[i-1]
            #integration
            for i in np.arange(0, N):
                du[i] = du[i]+ddu[i]*dt
                u[i] = u[i]+du[i]*dt
            #set to zero forces
            ddu=np.zeros(N)
            #dump to file
            if step%dump_interval == 0:
                filename=str(i_thread*Nrealisations + j)+'_'+str(int(step))+'.dump'
                path='dump/'+filename
                np.save(path, np.square(du))

        du2_averaged = du2_averaged + np.square(du)
        if (j+1)%50 == 0:
            logging.info('%4d-th thread completed %4d realisations\n', i_thread, j) 
    return du2_averaged/Nrealisations

def collect_result(result):
    global results
    results.append(result.tolist())

def average_results(N, Nrealisations, i_thread, step):
    averaged_dir='averaged'
    if not os.path.exists(averaged_dir):
        os.mkdir(averaged_dir)
    average=np.zeros(N)
    for j in np.arange(i_thread*Nrealisations, (i_thread+1)*Nrealisations):
        filename = str(j)+'_'+str(step)+'.dump.npy'
        path='dump/' + filename
        current_realisation = np.load(path)
        average=average+current_realisation
    average=average/Nrealisations
    logging.info('%d-th thread finished averaging  for %4d-th step\n', i_thread, step)
    return average

n_particles=401
n=np.arange(-(n_particles-1)/2, (n_particles-1)/2 + 1)
realisations=1000  #important: number of realisations must be dividible by number of cpus
dump_interval=100
dt=2*np.pi*1e-2
n_steps=2000
n_cpu=10 
pool=mp.Pool(n_cpu)
realisations_per_cpu=int(realisations/n_cpu)

for i in np.arange(0, n_cpu):
   pool.apply_async(run_simulations, args=(n_particles, realisations_per_cpu, n_steps, dt, i, dump_interval))
pool.close() 
pool.join()

for step in np.arange(0, n_steps, dump_interval):
	results=[]
	pool=mp.Pool(n_cpu)
	for i in np.arange(0, n_cpu):
		pool.apply_async(average_results, args=(n_particles, realisations_per_cpu, i, step), callback=collect_result)
	pool.close()
	pool.join()
	filename= str(step)+'.dump'
	path='averaged/'+filename
	np.savetxt(path, np.mean(results, axis=0), fmt='%1.9f', newline='\n')

os.system('python clean_dump.py')

from solver import * 
from generator import *
import numpy as np
import time

def preprocess(inp,lam,size):
    t00 = 0
    inp = pad_grids(inp,lam)
    inp = ( inp[[t00],:,:] - 500 ) / 250
    inp = inp.reshape(1,1,size,size,1)
    return inp


def recursive_prediction(model,inp,K,grid_size):
    '''
    model : Deep learning model
    inp   : Input solution for the model
    K     : iterations for the deep learning model
    '''
    solutions =np.zeros((1,K,grid_size,grid_size,1))
    solutions[:,[0]] = model.predict(inp)
    for ki in range(1,K): solutions[:,[ki]] = model.predict(solutions[:,[ki-1]])
    return solutions
        

def iterations_speedup_analysis(model,grid_size=12,P=10,K=10,max_iter=20) :
    '''
    grid_size : square grid size 
    P         : steps per deep learning prediction
    K         : total iterations for numerical solution = K * P
    '''   

    log={}
    error ={}

    mean,std=500,250 #standardization params
    bc1,bc2,bc3,bc4,ic0,lam= 600,500,194,248,254,0.27047  # problem params
    t00 = 0

    n = grid_size-2
    grid = generate_grid(n,bc=(bc1,bc2,bc3,bc4),ic = ic0)
    
    for ki in range(1,max_iter+1) :
    
        #numerical timings
        tic0=time.time() ; solution = solve(grid.copy(),iters=t00+P*ki,Lambda=lam,steps=True);toc0=time.time()
        tic1=time.time() ; prediction = recursive_prediction(model,preprocess(solution,lam,grid_size),ki,grid_size);toc1=time.time()
        print(f'Iterations = {ki*P} , grid size = {grid_size}x{grid_size}')
        
        log[ki*P]={'NumericalTime':(toc0-tic0)*1000 , 
              'PredictionTime':(toc1-tic1)*1000,
              'MAE':mae(solution[np.arange(P,ki*P+1,P)],(prediction[0,:,:,:,0]*250 )+ 500)}
        print(log[ki*P],'\n________________________________________________________')
        
    return solution,prediction,log


def gridsize_speedup_analysis(model,P=10,K=10,grids=[12]) :
    '''
    grid_size : square grid size 
    P         : steps per deep learning prediction
    K         : total iterations for numerical solution = K * P
    '''   

    log={}
    error ={}

    mean,std=500,250 #standardization params
    bc1,bc2,bc3,bc4,ic0,lam= 600,500,194,248,254,0.27047  # problem params
    t00 = 0

    for gi in grids :
        
        n = gi-2
        grid = generate_grid(n,bc=(bc1,bc2,bc3,bc4),ic = ic0)
    
        #numerical timings
        tic0=time.time() ; solution = solve(grid.copy(),iters=t00+P*K,Lambda=lam,steps=True);toc0=time.time()
        tic1=time.time() ; prediction = recursive_prediction(model,preprocess(solution,lam,gi),K,gi);toc1=time.time()
        print(f'Iterations = {K*P} , grid size = {gi}x{gi}')
        
        log[gi]={'NumericalTime':(toc0-tic0)*1000 , 
              'PredictionTime':(toc1-tic1)*1000,
              'MAE':mae(solution[np.arange(P,K*P+1,P)],(prediction[0,:,:,:,0]*250 )+ 500)}
        print(log[gi],'\n________________________________________________________')

    return solution,prediction,log

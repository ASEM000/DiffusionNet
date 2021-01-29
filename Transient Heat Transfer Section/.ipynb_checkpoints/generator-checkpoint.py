import numpy as np
import ray
from tqdm.notebook import tqdm
from solver import *

def pad_grids(grids,Lambda):
    grids[:,0,0]=grids[:,-1,0]=grids[:,0,-1]=grids[:,-1,-1]=Lambda * 1000
    return grids

@ray.remote
def solve_permutation(n,iters,permutation):
    Lambda,bc1,bc2,bc3,bc4,ic = permutation
    grid = generate_grid(n-2,bc=(bc1,bc2,bc3,bc4),ic=ic)
    ADIsoltuion = solve(grid.copy(),Lambda = Lambda,iters =iters ,steps=True)
    return  np.array(pad_grids(ADIsoltuion,Lambda)).reshape(ADIsoltuion.shape[0],ADIsoltuion.shape[1],ADIsoltuion.shape[2],1)

def generate_data(N,iters,permutations):
    '''
    Input : 
    N            : size of grid
    iters        : max iterations done by solver
    permutations : the solution parameters as a set of permutation (Lambda,bc1,..bc4,ic0)
    
    Output:
    solution with shape (iters+1,N,N)
    '''
    data = [(solve_permutation.remote(N,iters,i)) for i in (permutations) ]
    return np.array([ray.get(datalet) for datalet in (data)])
    

def generate_data_random_permutations(lR=(0,0.25),tR=(0,1000),batches =1,batch_size=32,seed =42,split=1 ):
    '''
    Input:
    *Lambda range
    *Temperature range
    *Size of data
    
    Output:
    *Generate an array of size with elements of (Lambda,bc1,..bc4,ic0)
    '''
    np.random.seed(seed);
    lr = np.random.randint(low = tR[0] , high = tR[1] ,size=(batches,batch_size,6)).astype('float')
    lr[:,:,0] = ((lR[1]-lR[0])*(lr[:,:,0]-tR[0]))/(tR[1]-tR[0])
    return lr


def generate_data_batches(N=50,
                          lR=(0,0.5),
                          tR=(0,100),
                          max_iters=10,
                          seed=42,
                          steps=1,
                          step_size=1,
                          batch_size=32,
                          batches=100,
                          progress=True,
                          key_bias =0,
                          save_file = None):
    '''
    return dictionary with key of the batch number
    '''
    if save_file is not None : 
        hf = h5py.File(save_file,'w')

    np.random.seed(seed)
    perms = generate_data_random_permutations(lR=lR,tR=tR,batch_size=batch_size,batches=batches,seed=seed)
    iters_list=  np.random.randint(low=step_size,high=max_iters-step_size*(steps-1)+1,size=batches)

    #scaling 
    mean = (tR[1]-tR[0])/2
    std  = mean/2
    data={}
    
    for batch_num in tqdm(range(batches)):
        
        iter_n = iters_list[batch_num]
        
        generated_data = generate_data(N,iter_n+steps*step_size,perms[batch_num])
        extract_index = np.arange(iter_n-step_size,iter_n+step_size*steps,step_size)
        generated_data = generated_data[:,extract_index,:,:,:]
        
        if save_file is None :
            data[f'{batch_num + key_bias}'] = (generated_data -mean) /std
        else : 
            hf.create_dataset(f'{batch_num}',data = data[batch_num] , compression ='gzip')
    
    if save_file is not None :hf.close()
    else : return data

    
def data_generator(data):
    '''
    input : data dictionary (batch number :5D tensor data)
    output: input , target values
    '''
    batches = len(data.keys())
    batch_size = len(data['0'])
    batch_counter= 0
    
    while True:
        x,y = data[f'{batch_counter}'][:,:-1,:,:,:],data[f'{batch_counter}'][:,-1:,:,:,:]

        batch_counter += 1
        yield x,y
        if batch_counter == batches:batch_counter = 0
            
def generate_training_validation(N,S,B=10_000):
    
    train_data_batches =generate_data_batches(N=N,
                                              lR=(0,1),
                                              tR=(0,1_000),
                                              max_iters=1000,
                                              seed=42,
                                              batch_size=32,
                                              step_size= S,
                                              steps=1,
                                              batches=int(B*0.9),
                                              progress=True)
    #                                           save_file = 'N=100x100 batch_size=64 batches=5000 max_iters=1000')

    train_data_batches_bias =generate_data_batches(N=N,
                                              lR=(0,1),
                                              tR=(0,1_000),
                                              max_iters=20,
                                              seed=42,
                                              batch_size=32,
                                              step_size= S,
                                              steps=1,
                                              batches =int(B*0.1),
                                              key_bias = int(B*0.9),
                                              progress = True)
    #                                           save_file = 'N=100x100 batch_size=64 batches=5000 max_iters=1000')

    validation_data_batches =generate_data_batches(N=N,
                                                  lR=(0,1),
                                                  tR=(0,1_000),
                                                  max_iters=1000,
                                                  seed=0,
                                                  batch_size=32,
                                                  step_size= S,
                                                  steps=1,
                                                  batches=50,
                                                  progress=True)
    
    return train_data_batches,train_data_batches_bias,validation_data_batches

import numpy as np
import h5py
from tqdm.notebook import tqdm

def generate_data_batches(input_data,
                          steps=1,
                          seed=42,
                          max_iters = 100,
                          batch_size=32,
                          batches=100,
                          progress=True,
                          step_size = 10,
                          key_bias=0,
                          save_file = None):
    '''
    return dictionary with key of the batch number
    '''
    
    np.random.seed(seed)
    iters_list=  np.random.randint(low=step_size,high=max_iters-step_size*(steps-1)+1,size=batches)

    #scaling 
    mean = 0 #(tR[1]-tR[0])/2
    std  = 1 #mean/2
    data={}
    
    for batch_num in tqdm(range(batches)):
        
        iter_n = iters_list[batch_num]
        extract_index = np.arange(iter_n-step_size,iter_n+step_size*steps,step_size)
        generated_data = input_data[batch_num:batch_num+batch_size,extract_index,:,:,:]

        if save_file is None :
            data[f'{batch_num + key_bias}'] = (generated_data) 
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
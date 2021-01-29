
from tensorflow.keras.models import Model  # machine learning library
from tensorflow.keras.optimizers import * # machine learning library
from tensorflow.keras.layers import * # machine learning library
import tensorflow

def conv(x,f,k):
    x = TimeDistributed(Conv2D(f,(k,k),strides=1,padding='same',kernel_initializer='glorot_uniform',activation=LeakyReLU()))(x)
    return x
    
def deconv(x,f,k):
    x = TimeDistributed(Conv2DTranspose(f,(k,k),strides=1,padding='same',kernel_initializer='glorot_uniform',activation=LeakyReLU()))(x)
    return x
        
def dense_block(tensor, f, r,k):
    for _ in range(r):
        x = conv(tensor, f=4*f, k=1)
        x = conv(x, f=f, k=k)
        tensor = Concatenate()([tensor, x])
    return tensor

def inv_dense_block(tensor, f, r,k):
    for _ in range(r):
        x = deconv(tensor, f=4*f, k=1)
        x = deconv(x, f=f, k=k)
        tensor = Concatenate()([tensor, x])
    return tensor

def transition(x,s):
    ff = int(tensorflow.keras.backend.int_shape(x)[-1] * 0.5)
    m0 = TimeDistributed(Conv2D(ff,(1,1),strides=2*s,padding='same',kernel_initializer='glorot_uniform',activation=LeakyReLU()))(x)
    return m0

def inv_transition(x,s):
    ff = int(tensorflow.keras.backend.int_shape(x)[-1] * 0.5)
    m0 = TimeDistributed(Conv2DTranspose(ff,(1,1),strides=2*s,padding='same',kernel_initializer='glorot_uniform',activation=LeakyReLU()))(x)
    return m0

def dfn():
    
    k=3
    s=1;
    LR=1e-4
    
    r1,r2,r3 = 2 , 4 ,8
    f0,f1,f2,f3 = 128,32,32,32
    l1,l2 = 128 ,64

    x = Input(shape=(None, None,None, 1))
    c0 = TimeDistributed(Conv2D(f0,(k,k),strides=1,padding='same',kernel_initializer='glorot_uniform',activation=LeakyReLU()))(x)
########################################################################################################    
    e1 = dense_block(c0,f1,r=r1,k=k);m1 = transition(e1,s)
    e2 = dense_block(m1,f2,r=r2,k=k);m2 = transition(e2,s)
    e3 = dense_block(m2,f3,r=r3,k=k);
########################################################################################################
    e = ConvLSTM2D(l1,(2,2),padding='same',return_sequences=True)(e3)
    b = ConvLSTM2D(l2,(2,2),padding='same',return_sequences=True)(e)
    d = ConvLSTM2D(l1,(2,2),padding='same',return_sequences=True)(b)
########################################################################################################
    d1 = inv_dense_block(d ,f3,r=r3,k=k);m1 = inv_transition(d1,s)
    d2 = inv_dense_block(m1,f2,r=r2,k=k);m2 = inv_transition(d2,s)
    d3 = inv_dense_block(m2,f1,r=r1,k=k);
########################################################################################################
    out = conv(d3,f=1,k=1)
    model = Model(x,out)
    optimizer = Adam(learning_rate=LR)
    model.compile(loss='mae',optimizer=optimizer,metrics=['mse'])
    return model
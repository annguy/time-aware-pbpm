
"""
This scripts help to get best parameters for each model from the history of hyperparameter tuning.

"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

"""
        GridSearch
        ......................................
        Finding the best set of hyperparameters for each model
        
        
        
       Variables
        -------------------
        
        HP_NUM_UNITS: list
            Number of LSTM units
            
        HP_DROPOUT: list
            Dropout rate
            
        HP_OPTIMIZER: list
            Optimizer
            
        HP_LEARNING_RATE: list
            Learning Rate
            
        data: list
            validation loss list
            
        label: list
            list of the hyperparameters
            
        name: str
            name of the data file 
            
        model_choice: str
            input for chosing model 
            
        num_units: int
            LSTM units parameter
            
        dropout_rate:float
            Dropoutrate
            
        optimizer:str
            Optimizer
            
        learning_rate:float
            learning_rate
            
        i: int
            helper variable
            
        df: pandas dataframe
            captures best hyperparameters
            
        fig: matplotlib object
            plot the dataframe
        
            
"""



HP_NUM_UNITS = [64, 100]
HP_DROPOUT = [0.0, 0.2]
HP_OPTIMIZER = ['nadam']
HP_LEARNING_RATE = [0.0001,0.0002,0.001,0.002,0.01]

data=[]
label=[]
name=input('Enter name of data file without extension: ')
print('Enter 1: CS Model, 2: TLSTM Model, 3: CS_TLSTM Model')
model_choice=input('Model Number: ')

for units in HP_NUM_UNITS:   
    for dropout_rate in (HP_DROPOUT):
        for optimizer in (HP_OPTIMIZER):
            try:
                for learning_rate in (HP_LEARNING_RATE):
                    filename=name+'dict_values(['+str(units)+', '+str(dropout_rate)+', '+ "'{}'".format(optimizer)+', '+str(learning_rate)+'])' #naming convention as per folder structure
                    with open('history/'+model_choice+'/'+filename,'rb') as fp:
                        hist=pickle.load(fp)
                        data.append(np.min(hist['val_loss']))
                        label.append(str(units)+', '+str(dropout_rate)+', '+ optimizer+', '+str(learning_rate))
                        print(str(units)+', '+str(dropout_rate)+', '+ optimizer+', '+str(learning_rate),' Min Val_loss',np.min(hist['val_loss']))
                        print('')
            except OSError:
                pass

i=np.argmin(data)
print('')
print('                          units, dropout_rate, optimizer, learning_rate', 'loss')                    
print('Best set of parameters is: ', label[i],np.min(data)) 


df=pd.DataFrame(data={'Units, Drop_out, Opti, Lr':label,'Min_val_loss':data})
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=df.values, colLabels=df.columns, loc='center',fontsize=40,)
fig.tight_layout()
plt.show()
fig.savefig('Results/'+name+'_Model_'+model_choice+".pdf", bbox_inches='tight')
                    

"""
This scripts help to generate plots from history of hyperparameter tuning.
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 

"""
        Plot_hyper
        ......................................
        Plotting the validation loss curves for each model
        
        
        
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
            
        data2: list
            training loss list
        
        length: int
            training length(#epochs)
        
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
            
      
        f: matplotlib object
            plot the validation loss
        
            
"""



HP_NUM_UNITS = [64, 100]
HP_DROPOUT = [0.0, 0.2]
HP_OPTIMIZER = ['nadam']
HP_LEARNING_RATE = [0.0001,0.0002,0.001,0.002,0.01]


name=input('Enter name of data file without extension: ')
print('Enter 1: CS Model, 2: TLSTM Model, 3: CS_TLSTM Model')
model_choice=input('Enter Model Number: ')
for units in HP_NUM_UNITS:   
    for dropout_rate in (HP_DROPOUT):
        f= plt.figure(figsize=[10,10])
        for optimizer in (HP_OPTIMIZER):
            for learning_rate in (HP_LEARNING_RATE):
                try:
                    filename=name+'dict_values(['+str(units)+', '+str(dropout_rate)+', '+ "'{}'".format(optimizer)+', '+str(learning_rate)+'])'
                    with open('history/'+model_choice+'/'+filename,'rb') as fp:
                        hist=pickle.load(fp)
                        data=hist['val_loss']
                        data2=hist['loss']
                        length=range(len(hist['loss']))
                        plot1,=plt.plot(length,data,label='Val'+str(units)+', '+str(dropout_rate)+', '+ optimizer+', '+str(learning_rate))
                        plot2,=plt.plot(length,data2,linestyle='dashed',label='Train'+str(units)+', '+str(dropout_rate)+', '+ optimizer+', '+str(learning_rate))
                except OSError:
                    pass
        plt.title('Hyperparameter Tuning_with model_'+model_choice+'_'+name,fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlabel('Number of Epochs', fontsize = 16)
        plt.ylabel('Validation_Loss', fontsize = 16)
        plt.legend(loc='upper right',fontsize=10) 
        plt.grid()
        plt.show()
        f.savefig('Plots_tuning/'+model_choice+'/HyperparameterTuning_'+name+'_'+str(units)+'_'+str(dropout_rate)+'_'+ optimizer+'.pdf')       
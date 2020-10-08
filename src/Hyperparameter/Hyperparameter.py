"""This Script allows to the user to do hyperparameter tuning for the different models"""

from src.Data.Datahandler import Datahandler
from src.Features.ComputeCW import ComputeCW
from src.Features.Preprocess import Preprocess
from src.Models.Model import *
import matplotlib.pyplot as plt




import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True) 
from tensorboard.plugins.hparams import api as hp
import pickle

class HyperparameterTune:
    """
    HyperparameterTune
    ............................
    Tuning the Hyperparameters
    
    
    Attributes
    -----------------         
        HP_NUM_UNITS:tensorboard.plugins.hparams.summary_v2.HParam
            Number of LSTM units
            
        HP_DROPOUT:tensorboard.plugins.hparams.summary_v2.HParam
            Dropout rate
            
        HP_OPTIMIZER:tensorboard.plugins.hparams.summary_v2.HParam
            Optimizer
            
        HP_LEARNING_RATE:tensorboard.plugins.hparams.summary_v2.HParam
            Learning Rate
            
        name: str
            name of the data file 
            
        model_choice: str
            input for chosing model to tune
    
    Methods
    ------------------
        param_tune()
            Tuning the models with Hyperparameters
            
        plot_func()
            plot the loss functions of the tuned models
        
    """


    def __init__(self,HP_NUM_UNITS = [],
                 HP_DROPOUT = [],
                 HP_OPTIMIZER = [],
                 HP_LEARNING_RATE = [],
                 name='',
                 model_choice=''):
    
        """
        
        HP_NUM_UNITS:tensorboard.plugins.hparams.summary_v2.HParam
            Number of LSTM units
            
        HP_DROPOUT:tensorboard.plugins.hparams.summary_v2.HParam
            Dropout rate
            
        HP_OPTIMIZER:tensorboard.plugins.hparams.summary_v2.HParam
            Optimizer
            
        HP_LEARNING_RATE:tensorboard.plugins.hparams.summary_v2.HParam
            Learning Rate
            
        name: str
            name of the data file 
            
        model_choice: str
            input for choosing model to tune
        """
        self.HP_NUM_UNITS = HP_NUM_UNITS
        self.HP_DROPOUT = HP_DROPOUT
        self.HP_OPTIMIZER = HP_OPTIMIZER
        self.HP_LEARNING_RATE = HP_LEARNING_RATE
        self.name=name
        self.model_choice=model_choice
    
    
    
    def param_tune(self,):
    
        """
        Tuning the hyperparameters

        ****Helper Variables****
        self.HP_NUM_UNITS:tensorboard.plugins.hparams.summary_v2.HParam
            Number of LSTM units
            
        self.HP_DROPOUT:tensorboard.plugins.hparams.summary_v2.HParam
            Dropout rate
            
        self.HP_OPTIMIZER:tensorboard.plugins.hparams.summary_v2.HParam
            Optimizer
            
        self.HP_LEARNING_RATE:tensorboard.plugins.hparams.summary_v2.HParam
            Learning Rate
            
        self.name: str
            name of the data file 
            
        self.model_choice: str
            input for chosing model to tune

        eventlog : str
            file to be processed
            
        F: Datahandler Obj
            File reading and processing
   
        divisor: float
            average time between current and first events
            
        divisor2: float
            average time between current and first events
        
        char_indices : dict
            ascii coded characters of the unique activities to integer indices
            
        indices_char: dict
            integer indices to ascii coded characters of the unique activities 
            
        target_char_indices: dict
            ascii coded characters of the target unique activities to integer indices
            (target includes one excess activity '!' case end)
            
        target_indices_char: dict
            integer indices to ascii coded characters of the target unique activities
            
        maxlen
            maximum length of cases
            
        chars:list 
            ascii coded characters of the activities.
            
        target_chars: list 
            ascii coded characters of the target activities.
           (target includes one excess activity '!' case end)
           
        X: ndarray
           training_set data
           
        y_a: list
            training_set labels for next activity
            
        y_t: list
            training_set list 
            
        METRIC_LOSS: str
            Keeping track of the model loss
      
        num_units: int
            LSTM units parameter
            
        dropout_rate:float
            Dropoutrate
            
        optimizer:str
            Optimizer
            
        learning_rate:float
            learning_rate
        
        cw: ComputeCW obj
        
        class_weights: dict
            class weights for activities
            
        hparams: dict
            hyperparameter for the run
        
        session_num: int
            session number of the run
        
        run_dir: str
            path to store tensorboard files
        
        run_name: str
            name of the tensorboard file 
        
        run_stat: str
            hyperparameters used
            
        hist: dict
            history of the training
            
        loss: list
            val_loss of the last epoch 
        
        
        """
    
        

        eventlog=input("Enter the name of the file to be executed from data folder: ")
        print('  ')
        
        

        #Reading the Data
        F = Datahandler()
        self.name=F.read_data(eventlog)

        #F.name is the name of the file
        #F.spamread is the data of the file


        spamreader,max_task = F.log2np()
        D=Preprocess()
        divisor,divisor2,divisor3 = D.divisor_cal(spamreader)
        maxlen,chars,target_chars,char_indices,indices_char,target_char_indices,target_indices_char= D.dict_cal()
        num_features = len(chars) + 5
        X, y_a, y_t, d_t = D.training_set(num_features)
            
        
        
        self.model_choice = input( "Please Select model: 1.CS, 2.TLSTM, 3.CS_TLSTM ")
        

        self.HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([64,100]))
        self.HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.2]))
        self.HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['nadam']))
        self.HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001,0.0002,0.001,0.002,0.01]))

        METRIC_LOSS = 'loss'
        path=os.path.abspath(os.curdir)
        
        with tf.summary.create_file_writer(path+'/logs_'+self.name+'/'+self.model_choice+'/').as_default():
            hp.hparams_config(
            hparams=[self.HP_NUM_UNITS,self.HP_DROPOUT, self.HP_OPTIMIZER,self.HP_LEARNING_RATE],
            metrics=[hp.Metric(METRIC_LOSS, display_name='Loss')],)
            
      
        #Training


        if self.model_choice == '1':
            cw=ComputeCW()
            class_weights=cw.compute_class_weight(F.spamread)
            print('class_weights are: ', class_weights)
            def run(run_dir,hparams):
                with tf.summary.create_file_writer(run_dir).as_default():
                    M=CSModel(maxlen,
                              max_task,
                              target_chars,
                              self.name,
                              num_features)
                    loss,hist=M.train(X,y_a,y_t, class_weights,
                                      hparams,self.HP_NUM_UNITS,self.HP_DROPOUT,
                                      self.HP_OPTIMIZER,self.HP_LEARNING_RATE)
                    tf.summary.scalar(METRIC_LOSS, loss, step=1)
                    return loss,hist



        elif self.model_choice == '2':
            def run(run_dir,hparams):
                with tf.summary.create_file_writer(run_dir).as_default():
                    M=ALL_TLSTM_Model(maxlen,
                                      max_task,
                                      target_chars,
                                      self.name,
                                      num_features)
                    loss,hist=M.train(X,d_t,y_a,y_t,hparams,self.HP_NUM_UNITS,self.HP_DROPOUT,
                                      self.HP_OPTIMIZER,self.HP_LEARNING_RATE)
                    tf.summary.scalar(METRIC_LOSS, loss, step=1)
                    return loss,hist

        elif self.model_choice == '3':
            cw=ComputeCW()
            class_weights=cw.compute_class_weight(F.spamread)
            print('class_weights are: ', class_weights)

            def run(run_dir,hparams):
                with tf.summary.create_file_writer(run_dir).as_default():
                    M=CS_TLSTM_Model(maxlen,
                                      max_task,
                                      target_chars,
                                      self.name,
                                      num_features)
                    loss,hist=M.train(X,d_t,y_a,y_t,class_weights,hparams,self.HP_NUM_UNITS,self.HP_DROPOUT,
                                      self.HP_OPTIMIZER,self.HP_LEARNING_RATE)
                    tf.summary.scalar(METRIC_LOSS, loss, step=1)
                    return loss,hist



        session_num = 0

        for num_units in (self.HP_NUM_UNITS.domain.values):
            for dropout_rate in (self.HP_DROPOUT.domain.values):
                for optimizer in self.HP_OPTIMIZER.domain.values:
                    for learning_rate in self.HP_LEARNING_RATE.domain.values: 
                        hparams = {
                          self.HP_NUM_UNITS:num_units,
                          self.HP_DROPOUT: dropout_rate,
                          self.HP_OPTIMIZER: optimizer,
                          self.HP_LEARNING_RATE:learning_rate,
                      }
                    #run_name = "run-%d" % session_num
                        run_stat={h.name: hparams[h] for h in hparams}
                        print('Session',session_num)
                        run_name = str(run_stat.values())
                    #print('--- Starting trial: %s' % run_name)
                        run_stat={h.name: hparams[h] for h in hparams}
                        print(run_stat)
                        loss,hist = run(path+'/logs_'+self.name+'/'+ self.model_choice+'/', hparams)
                        with open(path+'/history/'+self.model_choice+'/'+self.name+run_name, "wb") as fp:   
                            pickle.dump(hist, fp)
                        session_num += 1
                        
 

    def plot_func(self,):
        """
        Plots loss functions to a pdf file.   

        f: matplotlib obj
            ploting the loss 
            
        filename: str
            name of history of the loss to be plot
            
        data: list
            values to plot
        
        length: list
            number of epochs
        
        units: int
            LSTM units parameter
            
        dropout_rate:float
            Dropoutrate
            
        optimizer:str
            Optimizer
            
        learning_rate:float
            learning_rate
      
            
        hparams: dict
            hyperparameter for the run
                   
        hist: dict
            history of the training          
                   
        """ 
        path=os.path.abspath(os.curdir)
        for units in self.HP_NUM_UNITS.domain.values:            
            
            for dropout_rate in (self.HP_DROPOUT.domain.values): 
                f= plt.figure(figsize=[10,10])
                for optimizer in (self.HP_OPTIMIZER.domain.values):
                    for learning_rate in (self.HP_LEARNING_RATE.domain.values): 
                        filename=self.name\
                                 +'dict_values(['+str(units)\
                                 +', '+str(dropout_rate)+', '\
                                 + "'{}'".format(optimizer)+', '\
                                 +str(learning_rate)+'])'
                        with open('history/'+self.model_choice+'/'+filename,'rb') as fp:
                            hist=pickle.load(fp)
                            data=hist['val_loss']
                            data2=hist['loss']
                            length=range(len(hist['loss']))
                            plot1,=plt.plot(length,data,label='Val'
                                                              +str(units)
                                                              +', '+str(dropout_rate)+', '
                                                              + optimizer+', '+str(learning_rate))
                            plot2,=plt.plot(length,data2,linestyle='dashed',label='Train'
                                                                                  +str(units)+', '
                                                                                  +str(dropout_rate)
                                                                                  +', '+ optimizer
                                                                                  +', '+str(learning_rate))
                plt.title('Hyperparameter Tuning_with model_'+self.model_choice+'_'+self.name,fontsize = 16)
                plt.tick_params(axis='both', which='major', labelsize=15)
                plt.xlabel('Number of Epochs', fontsize = 16)
                plt.ylabel('Validation_Loss', fontsize = 16)
                plt.legend(loc='upper right',fontsize=10) 
                plt.grid()
                plt.show()
                
                f.savefig(path+'/Plots_tuning/'+self.model_choice+'/HyperparameterTuning_'
                          +self.name+'_'+str(units)
                          +'_'+str(dropout_rate)
                          +'_'+ optimizer+'.pdf')
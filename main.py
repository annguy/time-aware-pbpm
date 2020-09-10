"""This Script allows the user to run the Hyperparameter_Tuning, training, test, evaluations"""


##import Headers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from src.Data.Datahandler import Datahandler
from src.Features.ComputeCW import ComputeCW
from src.Features.Preprocess import Preprocess
from src.Hyperparameter.Hyperparameter import *
from src.Models.Model import *
from src.Models.Test import NextStep
from src.Evaluates.Evaluate import Nextstep_eval
from tensorboard.plugins.hparams import api as hp


"""
main.py
--------------
Complete Hyperparameter_Tuning, training, test and evaluation can be performed with this script

choice : str
    selection of task: hyperparameter tuning, training, testing or evaluation
        
""" 
#Task selection
choice=input('Enter one of the following choices: 0.Hyperparameter_Tuning, 1.Train, 2.Test, 3.Evaluate: ')
print('  ')

#Hyperparameter_Tuning    
"""        
    sure : str
        double confirmation for running hyperparameter tuning        
    h_t : object
        object of Hyperparameter class
        
    """
if choice == '0' or choice =='Hyperparameter_Tuning' or choice == '0.Hyperparameter_Tuning':
    sure=input('Hyperparameter Tuning can take long hours, press y to continue: ')
    
    if sure== 'y':
        h_t = HyperparameterTune()
        h_t.param_tune()
        h_t.plot_func()


#Training
    """
    ****Helper Variable****
    eventlog : str
        input data file
    name : str
        name of the datafile without extension
    num_features : int
        number of features
    F : object
        Datahandler object        
    spamreader: obj
            values of the pandas input table
    max_task: int
        number of unique tasks
    factor : double
        context normalisation factor     
    divisor: float
        average time between current and first events
    divisor2: float
        average time between current and first events
    divisor3 :float
        remaining time divisor    
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
    HP_NUM_UNITS: list
            Number of LSTM units        
    HP_DROPOUT: list
        Dropout rate        
    HP_OPTIMIZER: list
        Optimizer        
    HP_LEARNING_RATE: list
        Learning Rate
    model_choice : str
        Select model number to train
    M : obj
        Model Object
    cw : obj
        ComputeCW object
    class_weights : dict
        computed class weights
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

    """

if choice == '1' or choice =='Train' or choice == '1.Train':
    #Min Val Loss:
    #BPI12w: 64,0.0,nadam,0.01


    eventlog=input("Enter the name of the file to be executed from data\processed folder: ")
    print('  ')
    #Reading the Data
    F = Datahandler()
    name=F.read_data(eventlog)
    #F.name is the name of the file
    #F.spamread is the data of the file
    spamreader,max_task = F.log2np()
    D=Preprocess()
    divisor,divisor2,divisor3 = D.divisor_cal(spamreader)
    maxlen,chars,target_chars,char_indices,indices_char,target_char_indices,target_indices_char= D.dict_cal()
    num_features = len(chars)+5
    X,y_a,y_t,d_t=D.training_set(num_features)

    #Select best Hyperparamters for the datasets
    #Selectbest Hyperparameter_Tuning
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2]))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['nadam']))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.002]))
    model_choice = input( "Please Select model: 1.Class_Weighted, 2.All_TLSTM, 3.CS_TLSTM: ")
    #Training
    if model_choice == '1':
        cw = ComputeCW()
        class_weights = cw.compute_class_weight(F.spamread)
        print('class_weights are: ', class_weights)
        def run(hparams):
            M = CSModel(maxlen,
                        max_task,
                        target_chars,
                        name,
                        num_features)
            M.train(X, y_a, y_t, class_weights, hparams,
                    HP_NUM_UNITS, HP_DROPOUT,HP_OPTIMIZER, HP_LEARNING_RATE)

    elif model_choice == '2':
        def run(hparams):
            M=ALL_TLSTM_Model(maxlen,
                              max_task,
                              target_chars,
                              name,
                              num_features)
            M.train(X,d_t,y_a,y_t,
                    hparams,HP_NUM_UNITS,HP_DROPOUT,
                    HP_OPTIMIZER,HP_LEARNING_RATE)

    elif model_choice == '3':
        cw = ComputeCW()
        class_weights = cw.compute_class_weight(F.spamread)
        print('class_weights are: ', class_weights)
        def run(hparams):
            M = CS_TLSTM_Model(maxlen,
                                max_task,
                                target_chars,
                                name,
                                num_features)
            M.train(X, d_t, y_a, y_t, class_weights,
                    hparams, HP_NUM_UNITS, HP_DROPOUT,
                    HP_OPTIMIZER, HP_LEARNING_RATE)

    for num_units in (HP_NUM_UNITS.domain.values):
        for dropout_rate in (HP_DROPOUT.domain.values):
            for optimizer in HP_OPTIMIZER.domain.values:
                for learning_rate in HP_LEARNING_RATE.domain.values: 
                    hparams = {
                      HP_NUM_UNITS:num_units,
                      HP_DROPOUT: dropout_rate,
                      HP_OPTIMIZER: optimizer,
                      HP_LEARNING_RATE:learning_rate,
                  }
                
                    run_stat={h.name: hparams[h] for h in hparams}
                    run_name = str(run_stat.values())
                
                    print(run_stat)
                    run(hparams)

#Test
    """
    eventlog : str
        input data file
    name : str
        name of the datafile without extension
    num_features : int
        number of features
    F : object
        Datahandler object        
    spamreader: obj
            values of the pandas input table
    max_task: int
        number of unique tasks
    factor : double
        context normalisation factor     
    divisor: float
        average time between current and first events
    divisor2: float
        average time between current and first events
    divisor3 :float
        remaining time divisor    
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
    lines: list
        these are all the activity seq
    caseids: list
        caseid of the test sets
    lines_t:list
        time sequences (differences between the current and first)
    lines_t2: list
        time from case start
    lines_t3: list
        time features for midnight time
    Nxt: Object
        NextStep Object for next event related predictions
    Rmn: Object
        Remaining_Step Object for suffix and remaining case time
          
    """

elif choice == '2' or choice =='Test' or choice == '2.Test':
    eventlog=input("Enter the name of the file to be executed from data folder: ")
    print('  ')
    model_name=input("Enter the model path e.g model/sub_path/model_name.h5: ")
    print('  ')
    #Reading the Data
    F = Datahandler()
    name=F.read_data(eventlog)
    spamreader,max_task = F.log2np()
    D=Preprocess()
    divisor,divisor2,divisor3 = D.divisor_cal(spamreader)
    maxlen,chars,target_chars,char_indices,indices_char,target_char_indices,target_indices_char= D.dict_cal()
    lines,caseids,lines_t,lines_t2,lines_t3=D.test_set()
    num_features = len(chars)+5
#NextStep Prediction
    Nxt=NextStep(lines,caseids,lines_t,lines_t2,lines_t3,
                 maxlen,eventlog,chars,target_chars,divisor,
                 divisor2,target_indices_char,char_indices,
                 model_name,num_features)
    print('Running Test')
    Nxt.test()

#Evaluation
    """
    Eval: str
        evaluation choice for Nextstep or remaining step
    filename: str
        name of the file with test results
    Ev: obj
        Object of Nextstep_eval/Remainstep_eval class
    """
elif choice == '3' or choice =='Evaluate' or choice == '3.Evaluate':
    Eval=input('For evaluation of next activity and timestamp enter 1:  ')
    print('  ')
    if Eval == '1':
        try:
            filename = input("Enter the next_step file name of the format:Results/next_activity_time_filename.csv' : ") 
            print('  ')
            Ev=Nextstep_eval(filename)
            Ev.read()
            act_time=input("For time evaluations Enter 1, For activity related evaluations enter 2, Enter 3 for both: ")
            if act_time=='1':
                Ev.time()
            elif act_time=='2':
                Ev.activities()
            elif act_time=='3':
                Ev.time()
                Ev.activities()
        except IOError as e:
            errno, strerror = e.args
            print("I/O error({0}): {1}".format(errno,strerror))
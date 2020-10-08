"""This Script computes class weights of activity prediction for the different event logs only on the training set"""

from sklearn.utils import class_weight
import pandas as pd
import numpy as np


class ComputeCW:
    """
    ComputeCW
    ............
    Computes class weights of activity prediction for the different event logs


    Methods
    -----------
    compute_class_weight(self, spamread)
        computes the class weights

        Returns
        ----------
        class_weights: dict
        computed class_weights



    """


    def __init__(self,):
        """

        """
       
    def compute_class_weight(self, spamread):
        """
        Parameters
        -------------
        spamread: pd obj
           input data 

        ****Helper Variable****
        -----------------
        Train_1: pd obj
            first one third of the data
        Train_2:pd obj
            next one third of the data
        Test: pd obj
            test set or remaining one third 
            (Class weights are not  computed for this part)
        self.class_weights: dict
            computed class_weights


        Functions
        ----------
        add_last_row():
            adds the last row
            helper funtion
        
        
        Returns
        -------------
        self.class_weights: dict
            computed class_weights
        
        
        """
    
        def add_last_row(x):
        
            """
            adds the last activity as -1 to event balance class weights in one hot encoded targets for modelling 
            
            Parameters
            -----------

            x: pd obj
                input event log

            ****Helper Variable*****
            last_event: dict
                generated  to case end to balance class weights with target_char_indices
            d: pd obj
                helper variable

            Return
            ----------------
            df: pd obj
                output with last activity as -1.
            
            """
            last_event = {'CaseID': [''], 'ActivityID': [-1], 'CompleteTimestamp': [0]}
            d=pd.DataFrame(data=last_event)
            df = pd.concat([x,d], ignore_index=True)
            return df


        
        Train_1,Train_2,Test=np.array_split(spamread, 3)
        Train = pd.concat([Train_1,Train_2])
        Train=pd.DataFrame(Train.groupby('CaseID').apply(add_last_row).reset_index(drop=True))

        #The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001.
        self.class_weights = class_weight.compute_class_weight('balanced',np.unique(Train['ActivityID'].values),Train['ActivityID'].values.flatten())
        self.class_weights = dict(enumerate(self.class_weights))
        return self.class_weights
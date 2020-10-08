"""
This Script allows the user to test different models ,on the test_set i.e 1/3rd of the Dataset.
"""
from __future__ import division
import tensorflow as tf
import csv
import numpy as np
import distance
from jellyfish._jellyfish import damerau_levenshtein_distance
from sklearn import metrics
from datetime import datetime, timedelta
import os
from src.Models.TLSTM_layer import TLSTM_layer

"""
#CPU usage
config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
    
sess = tf.compat.v1.Session(config=config)
"""

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)         
    

class NextStep:
    """ 
    NextStep
    .........
     
    This class helps the user to predict next activity and timestamps.
    
    Attributes
    ----------------------
      
        divisor: float
            average time between current and first events
        divisor2: float
            average time between current and first events
        lines: list
            these are all the activity seq
        char_indices : dict
            ascii coded characters of the unique activities to integer indices
        indices_char: dict
            integer indices to ascii coded characters of the unique activities
        target_char_indices: dict
            ascii coded characters of the target unique activities to integer indices
            (target includes one excess activity '!' case end)
        target_indices_char: dict
            integer indices to ascii coded characters of the target unique activities
        lines: list
             ActivityIDs
        lines_t: list
            differences between two events
        lines_t2: list
            differences between the current and first of test_set
        lines_t3 : list
            Midnight time
        lines_t4 : list
            Day of the week
        one_ahead_gt : list
            helper variable to predict one ahead
        one_ahead_pred : list
            helper variable to predict one ahead
        two_ahead_gt : list
            helper variable to predict two ahead
        two_ahead_pred : list
            helper variable to predict two ahead
        three_ahead_gt :list
            helper variable to predict three ahead
        three_ahead_pred :list
            helper variable to predict three ahead

           
    Methods
    ------------------------
    encode(self,sentence, times, times3) : helper function
        encodes eventlog to Matrix, on which the Model can predict
        
    getSymbol(self,predictions): helper function
        encodes predicted values to ascii_coded activities
        
    test():prediction of the test set

    """







    def __init__(self,lines,caseids,lines_t,lines_t2,
                 lines_t3,maxlen,eventlog,chars,target_chars,
                 divisor,divisor2,target_indices_char,
                 char_indices,model_name,num_features,one_ahead_gt = [],
                 one_ahead_pred = [],two_ahead_gt = [],two_ahead_pred = [],
                 three_ahead_gt = [],three_ahead_pred = []):
        """
        Parameters
        -----------
    
        divisor: float
            average time between current and first events
        divisor2: float
            average time between current and first events
        lines: list
            these are all the activity seq
        char_indices : dict
            ascii coded characters of the unique activities to integer indices
        indices_char: dict
            integer indices to ascii coded characters of the unique activities 
        target_char_indices: dict
            ascii coded characters of the target unique activities to integer indices
            (target includes one excess activity '!' case end)
        target_indices_char: dict
            integer indices to ascii coded characters of the target unique activities    
        lines: list
             ActivityIDs 
        lines_t: list
            differences between two events 
        lines_t2: list
            differences between the current and first of test_set
        lines_t3 : list
            Midnight time
        lines_t4 : list
            Day of the week
        one_ahead_gt : list
            helper variable to predict one ahead
        one_ahead_pred : list
            helper variable to predict one ahead
        two_ahead_gt : list
            helper variable to predict two ahead
        two_ahead_pred : list
            helper variable to predict two ahead
        three_ahead_gt :list
            helper variable to predict three ahead
        three_ahead_pred :list
            helper variable to predict three ahead
    
        """
    
        self.lines=lines
        self.caseids=caseids
        self.lines_t=lines_t
        self.lines_t2=lines_t2
        self.lines_t3=lines_t3
        self.maxlen=maxlen
        self.target_chars=target_chars
        self.eventlog=eventlog
        self.chars=chars
        self.divisor=divisor
        self.divisor2=divisor2
        self.target_indices_char=target_indices_char
        self.char_indices=char_indices
        self.model_name=model_name
        self.one_ahead_gt = one_ahead_gt
        self.one_ahead_pred = one_ahead_pred
        self.two_ahead_gt = two_ahead_gt
        self.two_ahead_pred = two_ahead_pred
        self.three_ahead_gt = three_ahead_gt
        self.three_ahead_pred = three_ahead_pred
        self.num_features=num_features
        
      
     # define helper functions
    def encode(self,sentence, times, times3, num_features):
        """
        Encodes the test_set event log to a predictable shape
        
        Parameters
        ---------------------
        sentence:list
            activities of the events
        times: list
            time difference between current and previous event
        times3: list
            timedifference between first and current event
        num_features:int
            number of features

        *****Helper Variables****
        X : ndarray
            input for prediction
        leftpad: int
            helper
        times2 : float
            timedelta
        midnight: float
            midnight time
        timesincemidnight: float
            time difference after midnight
        multiset_abstraction: int
            helper variable
        self.char_indices: dict
            indices of the encoded activities
        self.divisor: float
            time avergage factor
        self.divisor2: float
            time average factor
        dts: numpy.ndarray
            time deltas for each sample left padded not scaled in seconds (samples, time_steps)

        """

        X = np.zeros((1, self.maxlen, num_features), dtype=np.float32)
        leftpad = self.maxlen-len(sentence)
        times2 = np.cumsum(times)
        pos=num_features-5
        for t, char in enumerate(sentence):
            midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = times3[t]-midnight
            #multiset_abstraction = Counter(sentence[:t+1])
            for c in self.chars:
                if c==char:
                    X[0, t+leftpad, self.char_indices[c]]=1
                    
            X[0, t+leftpad, pos]= t+1
            X[0, t+leftpad, pos+1] = times[t]/self.divisor
            X[0, t+leftpad, pos+2] = times2[t]/self.divisor2
            X[0, t+leftpad, pos+3] = timesincemidnight.seconds/86400
            X[0, t+leftpad, pos+4] = times3[t].weekday()/7

        dts = X[:, :, 2] * self.divisor
        return [X, dts]
        
        
    def getSymbol(self, predictions):
        """
        Encodes predicted values to ascii_coded activities
       
        Parameters
        -------------

        predictions : float
            model predictions

        ***Helper Variables*****
        maxPrediction : int
            helper variable
        self.target_indices_char: dict
            stores ascii_coded symbols for a prediction


        Return
        -----------------------------
         symbol: char
            ascii_coded characters of prediction

        """
   
        
        maxPrediction = 0
        symbol = ''
        i = 1
        for prediction in predictions:
            if(prediction>=maxPrediction):
                maxPrediction = prediction
                symbol = self.target_indices_char[i]
            i += 1
        return symbol    
  
        
        
        
    def test(self,):
        """
        Generates a file with predictions for next activities and time


        ***Helper Variables***
        predict_size : int
            number of predictions
        model : tf.keras.models
            trained models complete path
        path1: str
            complete path of the model
        filename: str
            local path and name of the model
        path: str
            local path of model
        model_type: str
            name of the model
        file_name: str
            name of the output file
        spamwriter: object
            csv writer object
        prefix_size:int
            size of eventlog prefix_size
        self.lines: list
            these are all the activity seq
        self.char_indices : dict
            ascii coded characters of the unique activities to integer indices
        self.indices_char: dict
            integer indices to ascii coded characters of the unique activities 
        self.target_char_indices: dict
            ascii coded characters of the target unique activities to integer indices
            (target includes one excess activity '!' case end)
        self.target_indices_char: dict
            integer indices to ascii coded characters of the target unique activities    
        self.lines: list 
             ActivityIDs 
        self.lines_t: list
            differences between two events 
        self.lines_t2: list
            differences between the current and first of test_set
        self.lines_t3 : list
            Midnight time
        self.lines_t4 : list
            Day of the week
        self.one_ahead_gt : list
            helper variable to predict one ahead
        self.one_ahead_pred : list
            helper variable to predict one ahead
        self.two_ahead_gt : list
            helper variable to predict two ahead
        self.two_ahead_pred : list
            helper variable to predict two ahead
        self.three_ahead_gt :list
            helper variable to predict three ahead
        self.three_ahead_pred :list
            helper variable to predict three ahead  
        cropped_line: list
            running activities while predictions
        cropped_times: list
            running time differences while predictions
        cropped_times3: list
            running time difference from case starting
        line: char
            activity items
        time: float
            time difference current and previous event
        times3: float
            time difference current and fisrt event
        ground_truth: char
            Ground truth activity
        ground_truth_t: float
            Groud truth time difference
        predicted: char
            predicted activity as a char
        predicted: list
            predicted time storing list
        y : dict
            all predctions
        y_char : float
            numerical prediction for activities
        y_t: float
            direct time prediction
        output: list
            complete list of output
        """
        
        # set parameters
        predict_size = 1
        
        # load model, set this to the model generated by train.py
        model = tf.keras.models.load_model(self.model_name,compile=False,custom_objects={"TLSTM_layer":TLSTM_layer})
        
          #name of the output
        path1, filename = os.path.split(self.model_name)
        path, model_type = os.path.split(path1)
        
        #name
        file_name=model_type+self.eventlog
        
        # make predictions
        with open('Results/1hotnext_activity_and_time_%s' % file_name, 'w',encoding="utf-8") as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["CaseID", "Prefix length", "Groud truth", "Predicted","Confidence", "Levenshtein", "Damerau", "Jaccard", "Ground truth times", "Predicted times", "RMSE", "MAE"])
            for prefix_size in range(2,self.maxlen):
                print(prefix_size)
                for line, caseid, times, times3 in zip(self.lines, self.caseids, self.lines_t, self.lines_t3):
                    times.append(0)
                    cropped_line = ''.join(line[:prefix_size])
                    
                    cropped_times = times[:prefix_size]
                    cropped_times3 = times3[:prefix_size]
                    
                    if '!' in cropped_line:
                        continue # make no prediction for this case, since this case has ended already
                    ground_truth = ''.join(line[prefix_size:prefix_size+predict_size])
                    ground_truth_t = times[prefix_size:prefix_size+predict_size]
                    predicted = ''
                    predicted_t = []
                    for i in range(predict_size):
                        if len(ground_truth)<=i:
                            continue
                        enc = self.encode(cropped_line, cropped_times, cropped_times3, self.num_features)
                        y = model.predict(enc, verbose=0)
                        y_char = y[0][0]
                        y_t = y[1][0][0]
                        prediction = self.getSymbol(y_char)   
                        confidence=np.round(np.max(y_char)*100)
                        
                        cropped_line += prediction
                        if y_t<0:
                            y_t=0
                        cropped_times.append(y_t)
                        y_t = y_t * self.divisor
                        cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                        predicted_t.append(y_t)
                        if i==0:
                            if len(ground_truth_t)>0:
                                self.one_ahead_pred.append(y_t)
                                self.one_ahead_gt.append(ground_truth_t[0])
                        if i==1:
                            if len(ground_truth_t)>1:
                                self.two_ahead_pred.append(y_t)
                                self.two_ahead_gt.append(ground_truth_t[1])
                        if i==2:
                            if len(ground_truth_t)>2:
                                self.three_ahead_pred.append(y_t)
                                self.three_ahead_gt.append(ground_truth_t[2])
                        if prediction == '!': # end of case was just predicted, therefore, stop predicting further into the future
                            print('! predicted, end case')
                            break
                        predicted += prediction
                    output = []
                    if len(ground_truth)>0:
                        output.append(caseid)
                        output.append(prefix_size)
                        output.append(str(ground_truth))
                        output.append(str(predicted))
                        output.append(confidence)
                        output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                        dls = 1 - (damerau_levenshtein_distance(str(predicted), str(ground_truth)) / max(len(predicted),len(ground_truth)))
                        if dls<0:
                            dls=0 # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                        output.append(dls)
                        output.append(1 - distance.jaccard(predicted, ground_truth))
                        output.append('; '.join(str(x) for x in ground_truth_t))
                        output.append('; '.join(str(x) for x in predicted_t))
                        if len(predicted_t)>len(ground_truth_t): # if predicted more events than length of case, only use needed number of events for time evaluation
                            predicted_t = predicted_t[:len(ground_truth_t)]
                        if len(ground_truth_t)>len(predicted_t): # if predicted less events than length of case, put 0 as placeholder prediction
                            predicted_t.extend(range(len(ground_truth_t)-len(predicted_t)))
                        if len(ground_truth_t)>0 and len(predicted_t)>0:
                            output.append('')
                            output.append(metrics.mean_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                            output.append(metrics.median_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                        else:
                            output.append('')
                            output.append('')
                            output.append('')
                        spamwriter.writerow(output)
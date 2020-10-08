"""
This script allows user to read CSV files which are prepared from anonymized structured eventlogs.
The data should have CaseID, ActivityID and Timestamp. Other resources are optional.
"""

#Importing Libraries

import pandas as pd
import os


class Datahandler:

    """
    Data handler Class

    ...

    This class takes the CSV files as input and returns data after preprocessing.


    Attributes
    --------------
    
        spamread: obj
            pandas object to store the input eventlog.
        
        name: str
            name of the file
        
        ext: str
            extension of the file
 
   Methods

    ---------------
        read_data(self, filename):  
            Reads pandas CSV files as per user defined name and path of the file from main.    
        
        log2np(self,):
           Finds out number of Unique Activities
           Returns an array of values for further preprocessing.
           
    """
   

    def __init__(self,spamread=0,
                 spamreader=0,
                 name='',
                 ext=''):

        self.spamread = spamread
        self.spamreader= spamreader
        self.name= name
        self.ext= ext
    
    
    #import anonymized data
    def read_data(self, filename):
        """This function reads the Data from a CSV file and returns the name of the file.
        
        Parameters
        --------------
        filename: str
            name.extension of the input file

        *****Helper Variable*****
        eventlog: str
            stores the filename

        self.spamread: obj
            pandas object to store the input eventlog.

        self.ext: str
            extension of the file

        path: str
            path of the file

        Returns
        ----------------------
        self.name: str
            name of the file
      
        """
        #Enter the name of the file be processed
        eventlog = filename
        self.name,self.ext=os.path.splitext(eventlog)

        #import Data from the data folder
        path=os.path.abspath(os.curdir)
        self.spamread=pd.read_csv(path+'\Data\processed\%s' %eventlog,error_bad_lines=False,delimiter=',',quotechar='|', index_col=False)
        
        return self.name


    # Preprocess the event log data to form sequences    
    def log2np(self):
        """This function converts the pandas dataframe into a numpy array like object and returns it.

        *****Helper Variable*****

        self.spamread: obj
            pandas object to store the input eventlog.

        Returns
        ------------
        spamreader: obj
            values of the pandas input table
        max_task: int
            number of unique tasks
        """
        #Number of Unique Tasks
        max_task=len(self.spamread['ActivityID'].unique())
        print('Number of Unique Activities',max_task)
        spamreader=self.spamread.values
        return spamreader,max_task
        
    
  





    
    
    
    


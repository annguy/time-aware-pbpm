"""This Script evaluates the outputs of the test results"""


##import Headers
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Nextstep_eval:
    """
    Nextstep_eval Class
    ...............................

    Evaluates and plots test results related to next activity and event completion time prediction
    .................................

    Attributes
    -------------------------

    filename: str
        name of the  complete file
    df: dataframe
        data of the test results
    name: str
        only name of the file

    Methods
    --------------
    read():
        Reads the data

    time():
        evaluates and plots event completion time related predictions

    activities():
            evaluates and plots net activity prediction related predictions
      
    """

    def __init__(self,filename,
                 df=0,
                 name='',
                 ext=''):
        """  
        Parameters
        -------------------------
        filename: str
            user input for name of the file
        df: dataframe
            data of the test results
        name: str
            only name of the file
        """
        self.filename=filename
        self.df=df
        self.name=name
        self.ext=ext
        self.name,self.ext=os.path.splitext(filename)
        
    def read(self,):
        """
       ***Helper Variable****
        self.df: dataframe
            data of the test results
        """
        self.df=pd.read_csv(self.filename,error_bad_lines=False,delimiter=',',quotechar='|', index_col=False,encoding = "ISO-8859-1")
        print(self.df.head())
        
    def time(self,):
        """
        ******Helper Variables*****
        avg: list
            helper variable to average the time error per prefix step
        cases: list
            stores  the case IDs per prefix step
        prefix: list
            list of length of prefix steps
        f: matplotlib object
            plot the figure
        i : int
            iteration variable
        x : dataframe
            helper
        time : float
            MAE time in days per prefix step
        error: float
            time error        
        """
        avg=[]
        cases=[]
        prefix=[]
        f= plt.figure(figsize=[10,5])
        for i in self.df['Prefix length'].unique():
            print('')
            print('Prefix_Length',i )
            x= self.df[self.df['Prefix length']==i]
            time=x['MAE']/86400   
            Total=x.count()[0]
            cases.append(Total)
            error=np.mean(time)
            avg.append(error)
            prefix.append(i)
            print('Number of Cases=', Total)
            print('Average Error=',error,'Days')
        print('')
        print('Overall Average Error in Days',np.mean(avg))
        print('Overall Weighted Average Erorr in Days',np.average(avg,weights=cases))
        plt.plot(prefix,avg,marker='o',markerfacecolor='r',markersize=12,label='Error plot')
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.title('Nextstep_time_eval',fontsize = 15)
        plt.xlabel("Prefix_Steps",fontsize = 15)
        plt.ylabel("Error in Days",fontsize = 15)
        plt.grid()
        plt.legend()
        plt.show()
        f.savefig(self.name+'nexttime.pdf')    
        
    def activities(self,):
        """
       ****Helper Variables*****
        avg : list
            helper variable to average the time error per prefix step
        cases : list
            stores  the case IDs per prefix step
        prefix : list
            list of length of prefix steps
        f: matplotlib object
            plot the figure
        i : int
            iteration variable
        x : dataframe
            helper
        Wrong : int
            count of prediction mismatch
        Correct : int
            count of correct predictions
        Total: int
            number of samples per prefix step
        Accuracy: float
            prediction accuracy in percentage
        """
    
        acc=[]
        cases=[]
        prefix=[]
        f= plt.figure(figsize=[10,5])
        for i in self.df['Prefix length'].unique():
            print('')
            print('Prefix_Length',i )
            x= self.df[self.df['Prefix length']==i]
            #Wrong=x[x['Groud truth']!=x['Predicted']].count()[0]
            Correct=x[x['Groud truth']==x['Predicted']].count()[0]
            Total=x.count()[0]
            cases.append(Total)
            Accuracy=(Correct/Total)*100
            acc.append(Accuracy)
            prefix.append(i)
            print('Number of Cases=', Total)
            print('Accuracy=',Accuracy,'%')
        print('')
        print('Overall Accuracy',np.average(acc))
        print('Overall Weighted Accuracy',np.average(acc,weights=cases))
        plt.plot(prefix,acc,marker='X',markerfacecolor='r',markersize=12,label='Accuracy plot')
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.title('Nextstep_activity_eval',fontsize = 15)
        plt.xlabel("Prefix_Steps",fontsize = 15)
        plt.ylabel("Accuracy",fontsize = 15)
        plt.grid()
        plt.legend()
        plt.show()
        f.savefig(self.name+'nextactivity.pdf')    

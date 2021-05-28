# Time Matters: Time-Aware LSTMs for Predictive Business Process Monitoring
This project was conducted at the [Machine Learning and Data Analytics Lab](https://www.mad.tf.fau.de/ ) (CS 14),
Department of Computer Science Friedrich-Alexander-University Erlangen-Nuremberg (FAU) 



## Citation and Contact
You find a PDF of the paper (accapted at [Ml4PM](http://ml4pm2020.di.unimi.it/) workshop @ ICPM2020)
[https://doi.org/10.1007/978-3-030-72693-5_9](https://doi.org/10.1007/978-3-030-72693-5_9).

If you use our work, please also cite the paper:
```
@inproceedings{Nguyen_Chatterjee_Weinzierl_Schwinn_Matzner_Eskofier_2021, 
author={Nguyen, An and Chatterjee, Srijeet and Weinzierl, Sven and Schwinn, Leo and Matzner, Martin and Eskofier, Bjoern},
title={Time Matters: Time-Aware LSTMs for Predictive Business Process Monitoring}, 
DOI={10.1007/978-3-030-72693-5_9}, 
booktitle={Process Mining Workshops}, 
publisher={Springer International Publishing}, 
year={2021}, 
pages={112–123}}
```
Watch the presentation on [Youtube](https://www.youtube.com/watch?v=LQNj3ZsQzWc&feature=youtu.be&t=1001&ab_channel=ICPM2020)

If you would like to get in touch, please contact [an.nguyen@fau.de](mailto:an.nguyen@fau.de).


## Abstract

> > Predictive business process monitoring (PBPM) aims to predict future process 
> > behavior during ongoing process executions based on event log data. Especially, 
> > techniques for the next activity and timestamp prediction can help to improve the 
> > performance of operational business processes. Recently, many PBPM solutions based
> > on deep learning were proposed by researchers. Due to the sequential nature of event 
> > log data, a common choice is to apply recurrent neural networks with long short-term memory (LSTM) cells. 
> > We argue, that the elapsed time between events is informative. However, current PBPM techniques
> > mainly use 'vanilla' LSTM cells and hand-crafted time-related control flow features. 
> > To better model the time dependencies between events, we propose a new PBPM technique based 
> > on time-aware LSTM (T-LSTM) cells. T-LSTM cells incorporate the elapsed time between consecutive
> > events inherently to adjust the cell memory. Furthermore, we introduce cost-sensitive learning
> > to account for the common class imbalance in event logs. Our experiments on publicly available
> > benchmark event logs indicate the effectiveness of the introduced techniques.
> > Getting Started


## Getting started

1. Clone this repo 

2. Create new environment using the environment.yml 
```bash
 'conda env create -f environment.yml' 
```
[Link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

3. Install src in the new environment 
```bash
'pip install -e.'
```

4. Hyperparameter_Tuning, training, test, evaluation
```bash
'python main.py' 
```

5. Best hyperparameters list for each model
```bash
'python Gridsearch.py' 
```

6. Validation loss plots for the complete hyperparameter tuning
```bash
'python Plot_hyper.py' 
```

Contributors
------
[An Nguyen](https://www.mad.tf.fau.de/person/an-nguyen/) 

[Srijeet Chatterjee](https://www.linkedin.com/in/srijeet-chatterjee-43845577/) 


## Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── history           <-  Saved history of training
    │   ├──	1
    │   ├── 2
    │   └── 3
	│
    ├── Model            <- Trained models serialized
	│   ├── CS
	│   ├── TLSTM	
	│   └── CS_TLSTM
    │
    ├── notebooks          <- Jupyter notebooks.  Training ,Test and Evaluation Demo
	│   ├──	1. Train
	│   ├── 2. Test
	│   └── 3. Evaluate
    │
    ├── Plots_tuning     <-  Plots after Hyperparameter tuning with Individaul model
    │
    ├── envtironment.yml   <- The requirements file for reproducing the analysis environment, e.g.
    │ 
    ├── main.py   <- Hyperparameter_Tuning, training, test, evaluation
    │
    ├── Gridsearch.py   <- Returns the best set of hyperparameters with Gridsearch
    │
    ├── Plot_hyper.py   <- Loss curves for the complete hyperparameter tuning process	
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
	│
    └ src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── Data           <- Scripts to download or generate data
	   │   ├── __init__.py    <- Makes src a Python module
       │   └── Datahandler.py
       │
       │
	   ├── Features       <- Scripts to turn raw data into features for modeling
	   │   ├── __init__.py    <- Makes src a Python module
	   │   ├── ComputeCw.py
       │   └── Preprocess.py	   
       │
       ├── Models         <- Scripts to train models and then use trained models to make  predictions
	   │   ├── __init__.py    <- Makes src a Python module               
       │   ├── Model.py
       │   └── Test.py
	   │   
       ├── Hyperparameter          <- Scripts to tune hyperparameters
       │   ├── __init__.py    <- Makes src a Python module
       │   └── Hyperparameter.py 	
       │
       └── Evaluates  <- Scripts to creates evaluation and visualizations of prediction results
	       ├── __init__.py    <- Makes src a Python module 
           └── Evaluate.py
    
   


--------




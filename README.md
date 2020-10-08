Time Matters: Time-Aware LSTMs forPredictive Business Process Monitoring
==============================
This project was conducted at the [Machine Learning and Data Analytics Lab](https://www.mad.tf.fau.de/ ) (CS 14),
Department of Computer Science Friedrich-Alexander-University Erlangen-Nuremberg (FAU) 



Getting Started

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
![alt text](https://code.siemens.com/shs-bda/pbpm/-/tree/github/notebooks/screen.png?raw=true)

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


Project Organization
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


This project is for simulating the interactions between SRB and gender income gap.


- /data: 							family survey data for estimating parameters of I-I model and I-E model
- /experiment: 					 experimental raw data
- /sa: 						results of sensitivity analysis

- agent.py: 						agent file
- sex_distortion.py: 				model file
- SA_OFAT.py: 						ofat analysis file
- SA_Sobol.py: 						sobol analysis file
- sex_ratio_bias_experiment.py: 		multi iterations experiment
- SRB_Distortion_notebook.ipynb:		integrated simulation work flow, including parameter estimation and simulation.


Running single iteration and observing running states, please open ipynb file and execute from top to bottom.

Running multi replicates and generate time series data of each iterations, please run 'python sex_ratio_bias_experiment.py'. Change parameters by editting python source file.

Running sensitivity analysis, please run 'python SA_OFAT' and 'python SA_Sobol'

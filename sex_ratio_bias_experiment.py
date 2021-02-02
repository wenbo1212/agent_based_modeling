from sex_distortion import SexDistortion
import sex_distortion as sd
import numpy as np
from mesa.batchrunner import BatchRunner

iim_param=[0.01394223,1.07057479]
iem_param=[ 0.28925115,1.43600609,-0.19679485]

#income_invest_model
iim=lambda income: 1/(iim_param[0]*income/100+iim_param[1])
#invest_edu_model
def iem(invest,gender):
    if(invest/100+1<0):
        print(invest)
    x=np.log(invest/100+1)
    return 1/(np.exp(-iem_param[0]*(x+iem_param[1]))+1)+iem_param[2]





fixed_params = {"iim":iim,
                "iem":iem,
                "width": 10,
               "height": 10,
               "base_population":100,
               "planning":3,
               "min_distance":0.67, #as previous setting
               "vision":1.1, #as previous
               }
uts=np.array([0.9,1.0,1.1])
variable_params={"initial_utility":uts}
iterations=1
batch_run = BatchRunner(SexDistortion,
                        fixed_parameters=fixed_params,
                        variable_parameters=variable_params,
                        iterations=iterations,
                        max_steps=10,
                        model_reporters={"male_income": sd.compute_male_income,
                              "female_income":sd.compute_female_income,
                              "male_count":sd.male_at_birth,
                              "female_count":sd.female_at_birth,
                              "avg_children":sd.avg_children,
                              "gender_ratio": sd.get_sex_ratio,
                              "pressure":sd.p_pressure
                             })
batch_run.run_all()
run_data =  batch_run.get_collector_model()
for k in run_data:
    run_data[k].to_csv("experiments/"+str(k)+".csv")

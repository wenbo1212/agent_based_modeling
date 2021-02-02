from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
import numpy as np
import math
import random
from agent import Male, Female

def compute_male_income(model):
    male_income=0
    male_count=0
    for agent in model.schedule.agents:
        if(not agent.sex and (not agent.marriage)):
            male_income+=agent.income
            male_count+=1
    if(male_count==0):
        return 0
    return male_income/male_count

def compute_female_income(model):
    female_income=0
    female_count=0
    for agent in model.schedule.agents:
        if(agent.sex and (not agent.marriage)):
            female_income+=agent.income
            female_count+=1
    if(female_count==0):
        return 0
    return female_income/female_count

def male_at_birth(model):

    return model.children_counter[0]



def female_at_birth(model):
    return model.children_counter[1]
    
def get_sex_ratio(model):

    if (model.children_counter[1] !=0):
        return model.children_counter[0]/model.children_counter[1]
    else:
        return 1
        
def avg_children(model):
    ch=0
    cc=1
    for agent in model.schedule.agents:
        if(agent.marriage):
            ch+=sum(agent.children)
            cc+=1
    return ch/cc

def p_pressure(model):
    m_count=0
    inv=0
    for agent in model.schedule.agents:
        if(agent.pp>0):
            m_count+=1
            inv+=agent.pp
    if(m_count==0):
        return 0
    return inv/m_count
    

class SexDistortion(Model):
    
    def __init__(self, iim,iem,height=10, width=10,
                 base_population=100,planning=3,
                 sex_ratio=1.0,
                 m_cost=0.5,
                 min_distance=0.67,
                 pressure=0.08,
                 fertility_control_coverage=0.1,  # income bias, p percent people hold (1-p) percent wealth, adjustable
                 vision=1.1, initial_utility=1.0):

        super().__init__()
        
        self.iim=iim
        self.iem=iem
        self.sex_ratio=sex_ratio
        self.social_level=5
        self.min_distance=min_distance
        self.fertility_control_coverage=fertility_control_coverage
        self.height = height
        self.marginal_cost=m_cost
        self.pressure=pressure
        self.width = width
        self.vision=vision
        self.schedule=RandomActivation(self)
        self.max_step=5
        self.grid = ContinuousSpace(self.width, self.height, torus=True)
        self.datacollector = DataCollector(
             model_reporters={"male_income": compute_male_income,
                              "female_income":compute_female_income,
                              "male_count":male_at_birth,
                              "female_count":female_at_birth,
                              "avg_children":avg_children,
                              "pressure":p_pressure,
                              "gender_ratio": get_sex_ratio
                             })
        self.org_ut=initial_utility
        discrimination=1-(1-initial_utility)/1000
        self.utility= (1,discrimination)
        #initial_utility
        self.planning=planning
        self.children_counter=[0,0]
        self.SRB_male=[]
        self.SRB_female=[]
        self.current_step=0
        self.initial_level_population=[0.31,0.22,0.17,0.2,0.10]
        self.initial_level_income=[20000,23000,25000,60000,100000]
        self.init_population_v1(base_population)
        self.level_income={}
        self.level_count={}
        self.level_bound={}
        self.initial_agents_income=[]

        # This is required for the datacollector to work
        self.running = True
        self.datacollector.collect(self)
        
    def init_population_v1(self, total):
        
        u=self.utility[1]/self.utility[0]
        #compute population detail by sex_ratio
        male_count=total*self.sex_ratio/(self.sex_ratio+1)
        female_count=total-male_count
        
        #approximate lorenz curve by staged poisson distribution
        distribution=[]
        for i in range(self.social_level):
            distribution.append(np.random.poisson(lam=self.initial_level_income[i]
                                                  ,size=math.ceil(male_count*self.initial_level_population[i])))
        X_male=np.concatenate(distribution)
        X_male=np.sort(X_male)
        pop_dist=male_count*np.array(self.initial_level_population)
        pop_dist=np.cumsum(pop_dist)
        slot_width=len(X_male)//self.social_level
        for i in range(len(X_male)):
            x = random.random()*(self.width)
            y = random.random()*(self.height)
            level=0
            for l,j in enumerate(pop_dist):
                if(i<j):
                    level=l
                    break
            self.new_agent(Male,(x,y),X_male[i],level)
            
        distribution=[]
        for i in range(self.social_level):
            distribution.append(np.random.poisson(lam=self.initial_level_income[i]*u
                                                  ,size=math.ceil(female_count*self.initial_level_population[i])))
        X_female=np.concatenate(distribution)
        X_female=np.sort(X_female)
        pop_dist=female_count*np.array(self.initial_level_population)
        pop_dist=np.cumsum(pop_dist)
        slot_width=len(X_female)//self.social_level
        for i in range(len(X_female)):
            x = random.random()*(self.width)
            y = random.random()*(self.height)
            level=0
            for l,j in enumerate(pop_dist):
                if(i<j):
                    level=l
                    break
            self.new_agent(Female,(x,y),X_female[i],level)
    
    #pos, income, residence, generation=1,clan=None,generation_in_clan=0, utility=1.0)
    def new_agent(self, agent_type, pos,income,level,study=0):
        '''
        Method that creates a new agent, and adds it to the correct scheduler.
        '''
        i=self.next_id()
        agent = agent_type(i, self, pos,income,level,study)

        self.grid.place_agent(agent, pos)
        self.schedule.add(agent)

    def remove_agent(self, agent):
        '''
        Method that removes an agent from the grid and the correct scheduler.
        '''
        self.grid.remove_agent(agent)
        self.schedule.remove(agent)

    def step(self):
        '''
        Method that calls the step method for each of the sheep, and then for each of the wolves.
        '''
        self.schedule.step()

        # Save the statistics
        self.datacollector.collect(self)
    
    def run_model(self, step_count=80):
        '''
        Method that runs the model for a specific amount of steps.
        '''
        for a in self.schedule.agents:
            self.initial_agents_income.append(a.income)
            
        for i in range(step_count):
            print(i)
            self.current_step=i
            print(self.datacollector.get_model_vars_dataframe()[-1:])
            self.step()
            print("children_counter is:",self.children_counter[0])
            self.SRB_male.append(self.children_counter[0])
            self.SRB_female.append(self.children_counter[1])
            print("sum is:",sum(self.SRB_male),sum(self.SRB_female))
            self.children_counter=[0,0]
from mesa import Agent
import random
import numpy as np
import scipy.optimize as opt
from scipy.optimize import Bounds
from functools import reduce

class RandomWalker(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.iim = model.iim
        self.iem = model.iem
        self.pos = pos

    def move(self,target_pos,vision):
        heading=self.model.grid.get_heading(self.pos,target_pos)
        next_pos=tuple(np.array(heading)*(1+vision/2))
        self.model.grid.move_agent(self,next_pos)
    def place(self,target_pos):
        self.model.grid.move_agent(self,target_pos)
        
    #generate a random position in a vision*vision square
    def _random_pos(self,vision):
        return tuple(np.array(self.pos)+(np.random.rand(2)-0.5)*2*vision)
    
    def random_move(self,vision):
        self.model.grid.move_agent(self,self._random_pos(vision))
        
class Individual(RandomWalker):
    def __init__(self, unique_id, model, pos, 
                 sex, income, level, study):
        super().__init__(unique_id, model, pos)
        self.sex=sex  #False:male, True:Female
        self.income=income
        self.marriage=False
        self.study=study
        self.investment=0 #total investment
        self.children=(0,0)  #(male_count, female_count)
        self.level=level #0->4 low->high
        self.vision=self.model.vision
        self.current_step=0
        self.pp=0


    def find_nearest(self,pos,neighbors):
        
        # find opposite gender satisfying following conditions
        op_gender=list(filter(lambda n:(n.sex^self.sex) 
                                and (not n.marriage)
                                and (np.abs(n.income-self.income)<self.income*0.3)
                                ,neighbors))
        incomes=list(map(lambda n:n.income,op_gender))
        if len(incomes)==0:
            return None
        i=np.argmax(incomes) 
        return op_gender[i]

    def get_neighbor_study(self,neighbors):
        male_study=[]
        female_study=[]
        for n in neighbors:
            if(n.study==0):
                continue
            each_inv=n.investment
            #male investments
            if(n.sex):
                female_study.append(n.study)
            else:
                male_study.append(n.study)
                
        return male_study,female_study
    def random_place(self):
        x = random.random()*(self.model.width)
        y = random.random()*(self.model.height)
        return x,y
    def competitiveness(self,self_study,neighbor_study,gender):
        avg=0
        if(gender):
            avg=(self_study+np.sum(neighbor_study[1]))/(len(neighbor_study[1])+1)
        else:
            avg=(self_study+np.sum(neighbor_study[0]))/(len(neighbor_study[0])+1)
        return self_study/avg

    #utility_function
    def individual_utility(self,income,son_num,daughter_num,neighbor_study,discrimination):
        expected_invest_ratio=self.iim(income)
        c_num=son_num+daughter_num
        n_num=len(neighbor_study[0])+len(neighbor_study[1])+c_num
        peer_pressure=(-np.log(c_num/n_num)+1)*self.model.pressure
        cps=len(neighbor_study[0])/n_num
        cpf=len(neighbor_study[1])/n_num
        self.pp=peer_pressure
        total_x=lambda x,c,frac:(x*(1-frac**c)/(1-frac))
        def target(x):
            u=0
            a=0.5
            investment=income*x
            life_quality=((2-2/(np.exp(-c_num*0.3*x)+1)-0.14)/(2-2/(np.exp(-c_num*0.3*expected_invest_ratio+1))-0.14))*100
#             life_quality=(1-total_x(invest_ratio,c_num,self.model.marginal_cost))/(1-expected_invest_ratio)
            if(son_num>0):
                son_study=self.iem(investment,False)
                us=(son_num*son_study*self.competitiveness(son_study,neighbor_study,False))
                if(discrimination<1):
                    dsc=1/discrimination
                    us*=dsc
                u+=us
            if(daughter_num>0):
                d_study=self.iem(investment,True)
                ud=(daughter_num*d_study*self.competitiveness(d_study,neighbor_study,True))
                if(discrimination>1):
                    ud*=discrimination
                u+=ud
            a=a/peer_pressure
            u*=100
            return -(u/c_num)**a*life_quality**(1-a)
 # minimum investment for each kid, adjustable
        res = opt.minimize(target, [0.2], bounds=[(0.01,0.3)] , method='TNC',options={'maxiter':20})
#         return total_x(res.x[0],c_num,self.model.marginal_cost),-target(res.x[0])
        return res.x[0],-target(res.x[0])
    def born(self,son,daughter):
        gender=False
        if(np.random.random()<0.5):
            son+=1
        else:
            daughter+=1
            gender=True
        return son,daughter,gender
    
    def get_neighbor_income(self,neighbor):
        incomes=[]
        incomes.append(self.income)
        for n in neighbor:
            incomes.append(n.income)
        return np.median(incomes)
    def _debug(self,investment,son,daughter,neighbor_study,u):
        print("debug test child:"+str((son,daughter)))
        std=self.iem(investment,False)
        dtd=self.iem(investment,True)
        print("debug study:"+str((std,dtd)))
        spc=self.competitiveness(std,neighbor_study,False)
        dpc=self.competitiveness(dtd,neighbor_study,True)
        print("debug compet:"+str((spc,dpc)))
        u1=(std*spc)
        u2=(dtd*dpc)
        print("debug ut_individual:"+str((u1,u2)))
        print("debug utility:"+str(u))
        
    def reproduce(self,neighbors):
        #step1 1st child
#         print("debug current id is:"+str(self.unique_id))
        control=random.random()<self.model.fertility_control_coverage
        son,daughter,_=self.born(0,0)
        neighbor_study=self.get_neighbor_study(neighbors)
        i0,u0=self.individual_utility(self.income,son,daughter,neighbor_study,self.model.utility[1])
#         print("debug current children:"+str((son,daughter)))
        while((son+daughter)<self.model.planning):
#             print("debug current utility:"+str((i0,u0)))
            i1,u1=self.individual_utility(self.income,son+1,daughter,neighbor_study,self.model.utility[1])
#             self._debug(i1*self.income,son+1,daughter,neighbor_study,u1)
            i2,u2=self.individual_utility(self.income,son,daughter+1,neighbor_study,self.model.utility[1])
#             self._debug(i2*self.income,son,daughter+1,neighbor_study,u2)
            if(np.mean([u1,u2])>u0):
                if(control):
                    if(u1>u2):
                        i0,u0=i1,u1
                        son+=1
                    elif(u1==u2):
                        if(random.random()<0.5):
                            i0,u0=i1,u1
                            son+=1
                        else:
                            i0,u0=i2,u2
                            daughter+=1
                    else:
                        i0,u0=i2,u2
                        daughter+=1
                else:
                    
                    son,daughter,g=self.born(son,daughter)
                    if(g):
                        i0,u0=i2,u2
                    else:
                        i0,u0=i1,u1
                
            else:
                break
#         print("debug decide:"+str((son,daughter)))
        self.investment=i0*self.income
        self.children=(son,daughter)
        self.model.children_counter[0]+=son
        self.model.children_counter[1]+=daughter
        each_invest=self.investment#/np.sum(self.children)
        son_study=self.iem(each_invest,False)
        daughter_study=self.iem(each_invest,True)
        sc=self.competitiveness(son_study,neighbor_study,False)
        dc=self.competitiveness(daughter_study,neighbor_study,True)
        baseline=self.get_neighbor_income(neighbors)
        male_income=baseline*sc
        female_income=baseline*dc
        if(self.model.utility[1]<0):
            dsc=1/self.model.utility[1]
            male_income*=dsc
        else:
            female_income*=(self.model.utility[1])
        male_level=self.level
        female_level=self.level
        for i in range(son):
            self.model.new_agent(Male,self.random_place(),male_income,male_level,son_study)
        for i in range(daughter):
            self.model.new_agent(Female,self.random_place(),female_income,female_level,daughter_study)
    
    def step(self):
        #define agent's life
        if(self.current_step>self.model.max_step):
            self.model.remove_agent(self)
            return 
        if(self.marriage):
            self.model.remove_agent(self)
            return
        self.current_step+=1
        neighbors=self.model.grid.get_neighbors(self.pos,self.vision,False)
        candidate=self.find_nearest(self.pos,neighbors)
        if(candidate is None):
            self.random_move(self.vision)
            return
        if(self.model.grid.get_distance(self.pos,candidate.pos)<self.model.min_distance):
            self.income+=candidate.income  #merge income
            self.marriage=True
            self.reproduce(neighbors)
            self.model.remove_agent(candidate) #remove the candidate because one representitive agent is enough for a family
        else:
            self.move(candidate.pos,self.vision)


class Male(Individual):
    def __init__(self,unique_id, model, pos, 
                 income, level, study):
        super().__init__(unique_id, model, pos, 
                 False, income, level, study)
                
        
class Female(Individual):
    def __init__(self,unique_id, model, pos, 
                 income, level, study):
        super().__init__(unique_id, model, pos, 
                 True, income, level, study)
    

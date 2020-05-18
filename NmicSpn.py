'''
n model with many servers -- immediate commitment version

routing and scheduling decisions are made by separated functions

routing: wwta / from rl
scheduling: cmu from rl
cost function: linear

revise function route_rl as remarked to enable rl control according to policy from nn:route_model
two possible actions to make:
    action 0: route incoming class 1 customer to queue 0
    action 1: route incoming class 1 customer to queue 1

revise function schedule_rl as remarked to enable rl control according to policy from nn:schedule_model
three possible actions to make:
    action 0: schedule a server from pool 2 for queue 2
    action 1: schedule a server from pool 2 for queue 3
    action 2: do nothing
pool 1 is simple non-idling FCFS

if use 'rl', please feed in nn_models into the corresponding arguments
'''

import numpy as np
import matplotlib.pyplot as plt

##################################################
# user input
##################################################


lambda_1,lambda_2 = (25,7)
mu_1,mu_2,mu_3 = (1,0.5,1)
N_1,N_2 = (20,20)

g_simulation_rounds = 50000

g_routing_policy_choice = 'wwta'
g_scheduling_policy_choice = 'cmu'
g_routing_nn_model = None
g_scheduling_nn_model = None
g_preemptive_choice = True



##################################################
# system internal info
##################################################

g_queue_lengths = np.zeros(3,dtype=int)
g_current_occupations = np.zeros(3,dtype=int) # how many servers each type of customers are currently occupying
g_event_indices = 0
g_current_cost = 0.0
g_avrg_cost = 0.0
g_current_cost_by_z1 = 0.0
g_avrg_cost_by_z1 = 0.0

g_lambda = np.array([lambda_1,lambda_2])
g_mu = np.array([mu_1,mu_2,mu_3])
g_unit_costs = np.array([3,1])
g_N = np.array([N_1,N_2])
g_total_max_rate = g_lambda.sum() + N_1*mu_1 + N_2*max(mu_2,mu_3)
g_ssc_cost_coef_z1 = g_unit_costs[0]+g_unit_costs[1]*mu_2

g_dynamic_data_bundle = {
    'queue lengths':g_queue_lengths,
    'occupations':g_current_occupations,
    'event number':g_event_indices,
    'current cost':g_current_cost,
    'average cost':g_avrg_cost,
}

g_static_data_bundle = {
    'arrival rates':g_lambda,
    'service rates':g_mu,
    'unit costs':g_unit_costs,
    'pool sizes':g_N,
    'uniformization factor':g_total_max_rate
}


g_policy_route = {
    'wwta':lambda current_queue,service_rates,jump_value,current_occupations,pool_size,route_model: route_wwta(current_queue,service_rates,jump_value,current_occupations,pool_size,route_model),
    'rl':lambda current_queue,service_rates,jump_value,current_occupations,pool_size,route_model:route_rl(current_queue,service_rates,jump_value,current_occupations,pool_size,route_model)
}
g_policy_schedule = {
    'cmu':lambda current_queue,service_rates,jump_real,current_occupations,pool_size,schedule_model,preemptive:schedule_cmu(current_queue,service_rates,jump_real,current_occupations,pool_size,schedule_model,preemptive),
    'rl':lambda current_queue,service_rates,jump_real,current_occupations,pool_size,schedule_model,preemptive:schedule_rl(current_queue,service_rates,jump_real,current_occupations,pool_size,schedule_model,preemptive) # this function assumes preemptive spn

}

def jump_next(dynamic_data,static_data): # next system jump event without routing/scheduling: arrive / service completion; does not alter the dynamic_data yet
    current_occupations = dynamic_data['occupations']
    total_rate = static_data['uniformization factor']
    lambda_val = static_data['arrival rates']
    mu_val = static_data['service rates']
    p_jump = np.concatenate((lambda_val,mu_val*current_occupations))/total_rate
    p_jump = np.concatenate((p_jump,np.array([1-p_jump.sum()])),0) # the probabilities for each events
    jump = np.zeros(5,dtype=int)
    event = np.random.choice(np.arange(0,6),p=p_jump)
    if event == 5:
        pass
    else:
        jump[event] = 1
    # cmf = np.cumsum(p_jump)
    # event = np.random.rand()
    # jump = np.zeros(5,dtype=int)
    # for index in range(len(jump)):
    #     if event<cmf[index]:
    #         jump[index] = 1
    #         return jump
    return jump

def route_wwta(current_queue,service_rates,jump_value,current_occupations,pool_size,route_model):
    jump_real = np.concatenate((np.zeros(2,dtype=int),np.array([jump_value[1]]),-jump_value[2:]),0)
    # routing decisions
    loads = current_queue/service_rates
    jump_real[int(loads[0]/service_rates[0]>(loads[1]+loads[2])/service_rates[1])] += 1
    return jump_real

def route_rl(current_queue,service_rates,jump_value,current_occupations,pool_size,route_model):
    jump_real = np.concatenate((np.zeros(2,dtype=int),np.array([jump_value[1]]),-jump_value[2:]),0)

    action_distr = np.array([.5,.5])

    '''
    fill in intermediate steps to extract the distribution for routing actions from nn:route_model
    '''

    action_sample = np.random.choice(np.arange(0,2),p=action_distr)
    # routing decisions
    jump_real[action_sample] = 1
    return jump_real

def schedule_cmu(current_queue,service_rates,jump_real,current_occupations,pool_size,schedule_model,preemptive):
    sche_jump = np.zeros(3,dtype=int)
    # scheduling decision
    if current_occupations[0] < pool_size[0] and (jump_real[0] == 1 or (jump_real[3] == -1 and current_queue[0]>current_occupations[0])): # event at server pool 1
        sche_jump[0] += 1
        return sche_jump

    if current_occupations[1] + current_occupations[2] < pool_size[1]: # event at server pool 2; servers not fully loaded
        if jump_real[1] == 1 or ((jump_real[4]+jump_real[5] == -1) and current_queue[1]>current_occupations[1]):
            sche_jump[1] += 1
        elif jump_real[2] == 1 or ((jump_real[4]+jump_real[5] == -1) and current_queue[2]>current_occupations[2]):
            sche_jump[2] += 1
    else: # preemption; notice if service completion take place at pool 2, then n_1+n_2<N_2 is guarrenteed
        if preemptive and jump_real[1] == 1 and current_occupations[2]>0:
            sche_jump += np.array([0,1,-1])
    return sche_jump

def schedule_rl(current_queue,service_rates,jump_real,current_occupations,pool_size,schedule_model,preemptive=True): # this function assumes preemptive spn
    sche_jump = np.zeros(3,dtype=int)

    if jump_real[0] == 1 or jump_real[3] == -1: # activity at pool 1
        sche_jump[0] += (current_queue[0] > current_occupations[0])
    elif current_queue[1] + current_queue[2] > current_occupations[1] + current_occupations[2]: # activity at pool 2
        while True:
            action_distr = np.array([1/3,1/3,1/3]) # action_distr should always be a numpy array with the illustrated dimensions

            '''
            fill in intermediate steps to extract the distribution for routing actions from nn:route_model
            '''
            
            action_sample = np.random.choice(np.arange(0,3),p=action_distr)
            if action_sample == 0 and (current_queue[1] == current_occupations[1] or current_occupations[1]==pool_size[1]): # action 0 infeasible
                action_distr[0] = 0
                psum = action_distr.sum()
                if psum == 0: # 
                    return sche_jump
                else:
                    action_distr = action_distr/psum
                    continue
            if action_sample == 1 and (current_queue[2] == current_occupations[2] or current_occupations[2]==pool_size[1]): # action 1 infeasible
                action_distr[1] = 0
                psum = action_distr.sum()
                if psum == 0: # 
                    return sche_jump
                else:
                    action_distr = action_distr/psum
                    continue
            if action_sample == 2: # notice action 2 is always feasible
                return sche_jump
            break
        # action 1 or action 2 is feasible
        sche_jump[action_sample+1] = 1
        sche_jump[2-action_sample] = - (current_occupations[1]+current_occupations[2]==pool_size[1]) # preempt if the requested server is serving the other queue

    return sche_jump

def control_next(dynamic_data,static_data,jump_value,route=g_routing_policy_choice,schedule=g_scheduling_policy_choice,route_model=g_routing_nn_model,schedule_model=g_scheduling_nn_model,preemptive=g_preemptive_choice):
    if jump_value.sum() == 0:
        return dynamic_data,static_data
    # static attributes
    service_rates = static_data['service rates']
    pool_size = static_data['pool sizes']
    # current dynamic attributes
    current_queue = dynamic_data['queue lengths']
    current_queue -= jump_value[2:]
    current_occupations = dynamic_data['occupations']
    current_occupations -= jump_value[2:]

    if jump_value[:1].sum() == 1:
        jump_real = g_policy_route[route](current_queue,service_rates,jump_value,current_occupations,pool_size,route_model)
    else:
        jump_real = np.concatenate((np.zeros(2,dtype=int),np.array([jump_value[1]]),-jump_value[2:]),0)

    sche_real = g_policy_schedule[schedule](current_queue,service_rates,jump_real,current_occupations,pool_size,schedule_model,preemptive=False)
    
    dynamic_data['queue lengths'] = current_queue + jump_real[0:3]
    dynamic_data['occupations'] = current_occupations + sche_real    

    return dynamic_data,static_data

def simulation(dynamic_data,static_data,simulation_rounds,route=g_routing_policy_choice,schedule=g_scheduling_policy_choice,route_model=g_routing_nn_model,schedule_model=g_scheduling_nn_model,preemptive=g_preemptive_choice):

    avg_cost_curve = np.zeros(simulation_rounds,dtype=float)
    q0_lengths = np.zeros(simulation_rounds,dtype=int)
    q1_lengths = np.zeros(simulation_rounds,dtype=int)
    q2_lengths = np.zeros(simulation_rounds,dtype=int)
    n1_occupat = np.zeros(simulation_rounds,dtype=int)
    n2_occupat = np.zeros(simulation_rounds,dtype=int)
    n3_occupat = np.zeros(simulation_rounds,dtype=int)

    unit_costs = static_data['unit costs']

    for i in range(simulation_rounds):
        # print('---------------------'+str(i)+'th'+'-----------------------')
        jump_value = jump_next(dynamic_data,static_data)
        # print('system event:',jump_value)
        dynamic_data,static_data = control_next(dynamic_data,static_data,jump_value,route,schedule,route_model,schedule_model,preemptive)
        dynamic_data['event number'] += 1
    
        j = dynamic_data['event number']

        dynamic_data['current cost'] += (unit_costs[0]*(jump_value[0] - jump_value[2] - jump_value[3]) + unit_costs[1]*(jump_value[1] - jump_value[4]))
        dynamic_data['average cost'] = dynamic_data['average cost']*(j-1)/j + dynamic_data['current cost']/j
    
        avg_cost_curve[i] = dynamic_data['average cost']

        q0_lengths[i],q1_lengths[i],q2_lengths[i] = dynamic_data['queue lengths']
        n1_occupat[i],n2_occupat[i],n3_occupat[i] = dynamic_data['occupations']
        
    return dynamic_data,static_data,avg_cost_curve,(q0_lengths,q1_lengths,q2_lengths,n1_occupat,n2_occupat,n3_occupat)








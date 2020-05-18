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
two possible actions to make:
    action 0: prioritize queue 2 at server pool 2
    action 1: prioritize queue 3 at server pool 3
pool 1 is simple non-idling FCFS

to use policies from nn trained by deep rl, please fill in correpsonding blocks at sampler
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
    'rl':lambda current_queue,service_rates,jump_value,current_occupations,pool_size,route_model:route_rl(current_queue,service_rates,jump_value,current_occupations,pool_size,route_model)
}
g_policy_schedule = {
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
    return jump

def sampler(current_queue,service_rates,current_occupations,route_model,schedule_model):
    r_action_distr = np.array([0,0])
    if route_model == None: # 'wwta' routing
        loads = current_queue/service_rates
        r_action_distr[int(loads[0]/service_rates[0]>(loads[1]+loads[2])/service_rates[1])] += 1
    else: # route_model is a NN
        pass
        '''
        fill in intermediate steps to extract the distribution for routing actions from nn:route_model
        '''
        
    r_action_sample = np.random.choice(np.arange(0,2),p=r_action_distr)

    s_action_distr = np.array([0,0])
    if schedule_model == None: # 'cmu' routing
        s_action_distr[0] = 1
    else: 
        '''
        fill in intermediate steps to extract the distribution for routing actions from nn:route_model
        '''
    s_action_sample = np.random.choice(np.arange(0,2),p=s_action_distr)

    return (r_action_sample,s_action_sample)


def route_rl(current_queue,service_rates,jump_value,current_occupations,pool_size,route_action_sample):
    jump_real = np.concatenate((np.zeros(2,dtype=int),np.array([jump_value[1]]),-jump_value[2:]),0)
    jump_real[route_action_sample] = 1
    return jump_real

def schedule_rl(current_queue,service_rates,jump_real,current_occupations,pool_size,schedule_action_sample,preemptive=True): # this function assumes preemptive spn
    sche_jump = np.zeros(3,dtype=int)
    action_sample = schedule_action_sample

    if current_occupations[0] < pool_size[0] and current_queue[0]>current_occupations[0]: # event at server pool 1
        sche_jump[0] = 1

    elif pool_size[1] > current_occupations[1] + current_occupations[2]: # activity at pool 2, pool 2 not fully loaded  
        if current_queue[action_sample+1] > current_occupations[action_sample+1]: # prioritized customer available
            sche_jump[action_sample+1] = 1
        elif current_queue[2-action_sample] > current_occupations[2-action_sample]: 
            sche_jump[2-action_sample] = 1 # serve if there are servers idle and customers waiting in the other queue
    elif preemptive: # prioritized queue has customers waiting and it is possible to make more room for the queue
        if current_queue[action_sample+1] > current_occupations[action_sample+1] and current_occupations[2-action_sample]>0:
            sche_jump[action_sample+1] = 1
            sche_jump[2-action_sample] = - 1 # preempt a server that is serving the other queue
    
    return sche_jump

def control_next(dynamic_data,static_data,jump_value,route_model=g_routing_nn_model,schedule_model=g_scheduling_nn_model,preemptive=g_preemptive_choice):
    # static attributes
    service_rates = static_data['service rates']
    pool_size = static_data['pool sizes']

    # current dynamic attributes
    current_queue = dynamic_data['queue lengths']
    current_occupations = dynamic_data['occupations']
    (r_action_sample,s_action_sample) = sampler(current_queue,service_rates,current_occupations,route_model,schedule_model)
    
    if jump_value.sum() == 0:
        return dynamic_data,static_data,(r_action_sample,s_action_sample)
    
    # current dynamic attributes
    current_queue -= jump_value[2:]
    current_occupations -= jump_value[2:]

    jump_real = np.concatenate((np.zeros(3,dtype=int),-jump_value[2:]),0)
    if jump_value[0] == 1:
        jump_real = route_rl(current_queue,service_rates,jump_value,current_occupations,pool_size,r_action_sample)
    elif jump_value[1]:
        jump_real[2] = 1

    current_queue += jump_real[0:3]

    sche_real = schedule_rl(current_queue,service_rates,jump_real,current_occupations,pool_size,s_action_sample,preemptive=False)
    
    dynamic_data['queue lengths'] = current_queue 
    dynamic_data['occupations'] = current_occupations + sche_real    

    return dynamic_data,static_data,(r_action_sample,s_action_sample)

def simulation(dynamic_data,static_data,simulation_rounds,route_model=g_routing_nn_model,schedule_model=g_scheduling_nn_model,preemptive=g_preemptive_choice):

    avg_cost_curve = np.zeros(simulation_rounds,dtype=float)
    q0_lengths = np.zeros(simulation_rounds,dtype=int)
    q1_lengths = np.zeros(simulation_rounds,dtype=int)
    q2_lengths = np.zeros(simulation_rounds,dtype=int)
    n1_occupat = np.zeros(simulation_rounds,dtype=int)
    n2_occupat = np.zeros(simulation_rounds,dtype=int)
    n3_occupat = np.zeros(simulation_rounds,dtype=int)

    r_actions = np.zeros(simulation_rounds,dtype=int)
    s_actions = np.zeros(simulation_rounds,dtype=int)

    unit_costs = static_data['unit costs']

    for i in range(simulation_rounds):
        # print('---------------------'+str(i)+'th'+'-----------------------')
        jump_value = jump_next(dynamic_data,static_data)
        # print('system event:',jump_value)
        dynamic_data,static_data,(r_actions[i],s_actions[i]) = control_next(dynamic_data,static_data,jump_value,route_model,schedule_model,preemptive)
        dynamic_data['event number'] += 1
    
        j = dynamic_data['event number']

        dynamic_data['current cost'] += (unit_costs[0]*(jump_value[0] - jump_value[2] - jump_value[3]) + unit_costs[1]*(jump_value[1] - jump_value[4]))
        dynamic_data['average cost'] = dynamic_data['average cost']*(j-1)/j + dynamic_data['current cost']/j
    
        avg_cost_curve[i] = dynamic_data['average cost']

        q0_lengths[i],q1_lengths[i],q2_lengths[i] = dynamic_data['queue lengths']
        n1_occupat[i],n2_occupat[i],n3_occupat[i] = dynamic_data['occupations']
        # print('----',i,'th ----')
        # print('jump',jump_value)
        # print('queue lengths:',dynamic_data['queue lengths'])
        # print('occupations:',dynamic_data['occupations'])
        
    return dynamic_data,static_data,avg_cost_curve,(q0_lengths,q1_lengths,q2_lengths,n1_occupat,n2_occupat,n3_occupat),(r_actions,s_actions)


#############################################################
# testing script
#############################################################
if __name__ == "__main__":
    dynamic_data,static_data,avg_cost_curve,states_traj,actions_traj=simulation(g_dynamic_data_bundle,g_static_data_bundle,g_simulation_rounds)

    x = np.arange(0, g_simulation_rounds, 1)
    plt.figure(1)
    plt.plot(x,avg_cost_curve,'k')
        # plt.plot(x,avg_cost_controled_curve,'c')
    plt.ylabel('average cost')
    plt.show()

    plt.figure(2)
    plt.plot(x,states_traj[0],'r',states_traj[1],'g',states_traj[2],'b')
        # plt.plot(x,avg_cost_controled_curve,'c')
    plt.ylabel('queue lengths')
    plt.show()

    plt.figure(3)
    plt.plot(x,states_traj[3],'r',states_traj[4],'g',states_traj[5],'b',states_traj[4]+states_traj[5],'k')
        # plt.plot(x,avg_cost_controled_curve,'c')
    plt.ylabel('occupations')
    plt.show()



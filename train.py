"""
Multiclass Queuing Network scheduling policy optimization using
Proximal Policy Optimization method with Approximating Martingale-Process variance reduction
PPO:
https://arxiv.org/abs/1707.06347 (by Schulman et al., 2017)
Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf (by Schulman et al., 2017)
Approximating Martingale-Process Method (by Henderson, Glynn, 2002):
https://web.stanford.edu/~glynn/papers/2002/HendersonG02.pdf
"""
import ray  # package for distributed computations
import numpy as np
from policy import Policy
from value_function import NNValueFunction
from utils import Logger, Scaler
import os
import argparse
import processingNetwork as pn
import random
import datetime
import copy
ray.init(temp_dir='/tmp/ray2')

MAX_ACTORS = 50  # max number of parallel simulations

def diag_dot(A, B):
    # returns np.diag(np.dot(A, B))
    return np.einsum("ij,ji->i", A, B)




def run_weights(network_id, weights_set, policy, scaler, cycles):

    if scaler.initial_states_procedure == 'previous_iteration':
        initial_state_0 = np.zeros(policy.get_obs_dim() + 1)
    else:
        initial_state_0 = np.zeros(policy.get_obs_dim())

    episodes = len(weights_set)

    remote_network = ray.remote(Policy)

    simulators = [remote_network.remote(policy.get_obs_dim(), policy.get_act_dim(), policy.get_kl_targ(),
                                        policy.get_hid1_mult()) for _ in range(episodes)]

    res = []

    ray.get([s.set_weights.remote(weights_set[i]) for i, s in enumerate(simulators)])

    scaler_id = ray.put(scaler)

    res.extend(ray.get([simulators[i].policy_performance_cycles.remote(network_id, scaler_id,   initial_state_0, i, batch_num = cycles)
                      for i in range(episodes)]))




    print('simulation is done')


    average_cost_set = np.zeros(episodes)
    ci_set = np.zeros(episodes)


    for i in range(episodes):

        average_cost_set[res[i][1]] = res[i][0]
        ci_set[res[i][1]] = res[i][2]

    print('Average cost: ', average_cost_set)
    print('CI: ', ci_set)
    return average_cost_set, ci_set



def run_policy(network_id, policy, scaler, logger, gamma,
               policy_iter_num, skipping_steps, cycles_num, episodes, time_steps):
    """
    Run given policy and collect data
    :param network_id: queuing network structure and first-order info
    :param policy: queuing network policy
    :param scaler: normalization values
    :param logger: metadata accumulator
    :param gamma: discount factor
    :param policy_iter_num: policy iteration
    :param skipping_steps: number of steps when action does not change  ("frame-skipping" technique)
    :param episodes: number of parallel simulations (episodes)
    :param time_steps: max time steps in an episode
    :return: trajectories = (states, actions, rewards)
    """

    total_steps = 0
    action_optimal_sum = 0
    total_zero_steps = 0

    burn = 1

    scale, offset = scaler.get()

    '''
    initial_states_set = random.sample(scaler.initial_states, k=episodes)
    trajectories, total_steps, action_optimal_sum, total_zero_steps, array_actions = policy.run_episode(ray.get(network_id), scaler, time_steps, cycles_num, skipping_steps,  initial_states_set[0])
    '''

    #### declare actors for distributed simulations of a current policy#####
    remote_network = ray.remote(Policy)
    simulators = [remote_network.remote(policy.get_obs_dim(),policy.get_act_dim(), policy.get_kl_targ(), policy.get_hid1_mult()) for _ in range(MAX_ACTORS)]
    actors_per_run = episodes // MAX_ACTORS # do not run more parallel processes than number of cores
    remainder = episodes - actors_per_run * MAX_ACTORS
    weights = policy.get_weights()  # get neural network parameters
    ray.get([s.set_weights.remote(weights) for s in simulators]) # assign the neural network weights to all actors
    ######################################################



    ######### save neural network parameters to file ###########
    file_weights = os.path.join(logger.path_weights, 'weights_'+str(policy_iter_num)+'.npy')
    np.save(file_weights, weights)
    ##################

    scaler_id = ray.put(scaler)
    initial_states_set = random.sample(scaler.initial_states, k=episodes)  # sample initial states for episodes

    ######### policy simulation ########################
    accum_res = []  # results accumulator from all actors
    trajectories = []  # list of trajectories
    for j in range(actors_per_run):
        accum_res.extend(ray.get([simulators[i].run_episode.remote(network_id, scaler_id, time_steps, cycles_num,
                                skipping_steps,  initial_states_set[j*MAX_ACTORS+i]) for i in range(MAX_ACTORS)]))
    if remainder>0:
        accum_res.extend(ray.get([simulators[i].run_episode.remote(network_id, scaler_id, time_steps, cycles_num,
                                skipping_steps, initial_states_set[actors_per_run*MAX_ACTORS+i]) for i in range(remainder)]))
    print('simulation is done')

    for i in range(len(accum_res)):
        trajectories.append(accum_res[i][0])
        total_steps += accum_res[i][1]  # total time-steps
        action_optimal_sum += accum_res[i][2]  # absolute number of actions consistent with the "optimal policy"
        total_zero_steps += accum_res[i][3]  # absolute number of states for which all actions are optimal
    #################################################


    optimal_ratio = action_optimal_sum / (total_steps * skipping_steps)  # fraction of actions that are optimal
    # fraction of actions that are optimal excluding transitions when all actions are optimal
    pure_optimal_ratio = (action_optimal_sum - total_zero_steps)/ (total_steps * skipping_steps - total_zero_steps)

    average_reward = np.mean(np.concatenate([t['rewards'] for t in trajectories]))


    #### normalization of the states in data ####################
    unscaled = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    if gamma < 1.0:
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]


    else:
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]
            z = t['rewards'] - average_reward
            t['rewards'] = z
    ##################################################################





    scaler.update_initial(np.hstack((unscaled, np.zeros(len(unscaled))[np.newaxis].T)))

    ########## results report ##########################
    print('Average cost: ',  -average_reward)

    logger.log({'_AverageReward': -average_reward,
                'Steps': total_steps,
                'Zero steps':total_zero_steps,
                '% of optimal actions': int(optimal_ratio * 1000) / 10.,
                '% of pure optimal actions': int(pure_optimal_ratio * 1000) / 10.,
    })
    ####################################################
    return trajectories

def add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration):
    """
    compute value function for further training of Value Neural Network
    :param trajectory: simulated data
    :param network: queuing network
    :param policy: current policy
    :param gamma: discount factor
    :param lam: lambda parameter in GAE
    :param scaler: normalization values
    """
    start_time = datetime.datetime.now()
    for trajectory in trajectories:

        if iteration!=1:

            values = trajectory['values']
            observes = trajectory['observes']
            unscaled_obs = trajectory['unscaled_obs']


            ###### compute expectation of the value function of the next state ###########
            probab_of_actions = policy.sample(observes) # probability of choosing actions according to a NN policy

            distr = np.array(probab_of_actions[0].T)
            for ar_i in range(1, network.stations_num):
                distr = [a * b for a in distr for b in np.array(probab_of_actions[ar_i].T)]

            distr = np.array(distr).T
            distr = distr / np.sum(distr, axis=1)[:, np.newaxis]  # normalization

            action_array = network.next_state_prob(unscaled_obs) # transition probabilities for fixed actions



            # expectation of the value function for fixed actions
            value_for_each_action_list = []
            for act in action_array:
                value_for_each_action_list.append(diag_dot(act, trajectory['values_set'].T))
            value_for_each_action = np.vstack(value_for_each_action_list)

            P_pi = diag_dot(distr, value_for_each_action)  # expectation of the value function
            ##############################################################################################################

            # # expectation of the value function w.r.t the actual actions in data
            # distr_fir = np.eye(len(network.actions))[trajectory['actions_glob']]
            #
            # P_a = diag_dot(distr_fir, value_for_each_action)


            # td-error computing
            tds_pi = trajectory['rewards'] - values + gamma*P_pi[:, np.newaxis]
            #tds_pi = trajectory['rewards'] # no control variate






            # value function computing for futher neural network training
            #TODO: ensure that gamma<1 works
            if gamma < 1:
                #advantages = discount(x=tds_pi,   gamma=lam*gamma, v_last = tds_pi[-1]) - tds_pi + tds_a   # advantage function
                disc_sum_rew = discount(x=tds_pi,   gamma= lam*gamma, v_last = tds_pi[-1]) + values
            else:
                #advantages = relarive_af(unscaled_obs, td_pi=tds_pi, td_act=tds_a, lam=lam)  # advantage function
                disc_sum_rew = relarive_af(unscaled_obs, td_pi=tds_pi, lam=lam) + values # value function
                #disc_sum_rew = relarive_af(unscaled_obs, td_pi=tds_pi, lam=lam) # value function -- No CV

        else:
            if gamma < 1:
                #advantages = discount(x=tds_pi,   gamma=lam*gamma, v_last = tds_pi[-1]) - tds_pi + tds_a   # advantage function
                disc_sum_rew = discount(x=trajectory['rewards'],   gamma= gamma, v_last = trajectory['rewards'][-1])
            else:

                #advantages = relarive_af(unscaled_obs, td_pi=tds_pi, td_act=tds_a, lam=lam)  # advantage function
                disc_sum_rew = relarive_af(trajectory['unscaled_obs'], td_pi=trajectory['rewards'], lam=1)  # advantage function


        #trajectory['advantages'] = np.asarray(advantages)
        trajectory['disc_sum_rew'] = disc_sum_rew


    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('add_disc_sum_rew time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

    burn = 1

    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])
    if iteration ==1:
        scaler.update(np.hstack((unscaled_obs, disc_sum_rew)))
    scale, offset = scaler.get()
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    disc_sum_rew_norm = (disc_sum_rew - offset[-1]) * scale[-1]
    if iteration ==1:
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]

    return observes, disc_sum_rew_norm

def discount(x, gamma, v_last):
    """ Calculate discounted forward sum of a sequence at each point """
    disc_array = np.zeros((len(x), 1))
    disc_array[-1] = v_last
    for i in range(len(x) - 2, -1, -1):
        if x[i+1]!=0:
            disc_array[i] = x[i] + gamma * disc_array[i + 1]

    return disc_array


def relarive_af(unscaled_obs, td_pi,  lam):
    # return advantage function
    disc_array = np.copy(td_pi)
    sum_tds = 0
    for i in range(len(td_pi) - 2, -1, -1):
        if np.sum(unscaled_obs[i+1]) != 0:
            sum_tds = td_pi[i+1] + lam * sum_tds
        else:
            sum_tds = 0
        disc_array[i] += sum_tds

    return disc_array



def add_value(trajectories, val_func, scaler, possible_states):
    """
    # compute value function from the Value Neural Network
    :param trajectory_whole: simulated data
    :param val_func: Value Neural Network
    :param scaler: normalization values
    :param possible_states: transitions that are possible for the queuing network
    """
    start_time = datetime.datetime.now()
    scale, offset = scaler.get()


    # approximate value function for trajectory_whole['unscaled_obs']
    for trajectory in trajectories:
        values = val_func.predict(trajectory['observes'])
        trajectory['values'] = values / scale[-1] + offset[-1]

        # approximate value function of the states where transitions are possible from trajectory_whole['unscaled_obs']
        values_set = np.zeros(( len(possible_states)+1, len(trajectory['observes'])))

        new_obs = (trajectory['unscaled_last'] - offset[:-1]) * scale[:-1]
        values = val_func.predict(new_obs)
        values = values / scale[-1] + offset[-1]
        values_set[-1] = np.squeeze(values)

        for count, trans in enumerate(possible_states):
            new_obs =(trajectory['unscaled_last'] + trans - offset[:-1]) * scale[:-1]
            values = val_func.predict(new_obs)
            values = values / scale[-1] + offset[-1]
            values_set[count] = np.squeeze(values)

        trajectory['values_set'] = values_set.T


    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('add_value time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

def build_train_set(trajectories, gamma, scaler):
    """
    # data pre-processing for training
    :param trajectory_whole:  simulated data
    :param scaler: normalization values
    :return: data for further Policy and Value neural networks training
    """



    for trajectory in trajectories:
        values = trajectory['values']

        unscaled_obs = trajectory['unscaled_obs']


        ###### compute expectation of the value function of the next state ###########


        action_array = network.next_state_prob(unscaled_obs) # transition probabilities for fixed actions



        # expectation of the value function for fixed actions
        value_for_each_action_list = []
        for act in action_array:
            value_for_each_action_list.append(diag_dot(act, trajectory['values_set'].T))
        value_for_each_action = np.vstack(value_for_each_action_list)


        ##############################################################################################################

        # # expectation of the value function w.r.t the actual actions in data
        distr_fir = np.eye(len(network.actions))[trajectory['actions_glob']]

        P_a = diag_dot(distr_fir, value_for_each_action)

        advantages = trajectory['rewards'] - values +gamma*P_a[:, np.newaxis]# gamma * np.append(values[1:], values[-1]), axis=0)  #


        trajectory['advantages'] = np.asarray(advantages)




    start_time = datetime.datetime.now()
    burn = 1


    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])

    scale, offset = scaler.get()
    actions = np.concatenate([t['actions'][:-burn] for t in trajectories])
    advantages = np.concatenate([t['advantages'][:-burn] for t in trajectories])
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages



    # ########## averaging value function estimations over all data ##########################
    # states_sum = {}
    # states_number = {}
    # states_positions = {}
    #
    # for i in range(len(unscaled_obs)):
    #     if tuple(unscaled_obs[i]) not in states_sum:
    #         states_sum[tuple(unscaled_obs[i])] = disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] = 1
    #         states_positions[tuple(unscaled_obs[i])] = [i]
    #
    #     else:
    #         states_sum[tuple(unscaled_obs[i])] +=  disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] += 1
    #         states_positions[tuple(unscaled_obs[i])].append(i)
    #
    # for key in states_sum:
    #     av = states_sum[key] / states_number[key]
    #     for i in states_positions[key]:
    #         disc_sum_rew[i] = av
    # ########################################################################################
    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('build_train_set time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')
    return observes,  actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    # metadata tracking

    time_total = datetime.datetime.now() - logger.time_start
    logger.log({'_mean_act': np.mean(actions),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode,
                '_time_from_beginning_in_minutes': int((time_total.total_seconds() / 60) * 100) / 100.
                })

# TODO: check shadow name
def main(network_id, num_policy_iterations, gamma, lam, kl_targ, batch_size, hid1_mult, episode_duration, cycles_num,
         clipping_parameter, skipping_steps, initial_state_procedure):
    """
    # Main training loop
    :param: see ArgumentParser below
    """


    obs_dim = ray.get(network_id).buffers_num
    act_dim = ray.get(network_id).action_size_per_buffer
    now = datetime.datetime.utcnow().strftime("%b-%d_%H-%M-%S")  # create unique directories
    time_start= datetime.datetime.now()
    logger = Logger(logname=ray.get(network_id).network_name, now=now, time_start=time_start)


    scaler = Scaler(obs_dim + 1, initial_state_procedure)
    val_func = NNValueFunction(obs_dim, hid1_mult) # Value Neural Network initialization
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult) # Policy Neural Network initialization




    # initialize as a random proportional policy
    # scale, offset = scaler.get()
    # initial_states_set = random.sample(scaler.initial_states, k=1)
    # trajectories, total_steps, action_optimal_sum, total_zero_steps, action_distr = \
    #     policy.run_episode(ray.get(network_id), scaler, episode_duration, skipping_steps, initial_states_set[0],
    #                        rpp=True)
    # state_input = (trajectories['unscaled_obs'] - offset[:-1]) * scale[:-1]
    # policy.initilize_rpp(state_input, action_distr)



    ############## creating set of initial states for episodes in simulations##########################
    if initial_state_procedure=='empty':
        x = np.zeros((50000, obs_dim), dtype= 'int8')
        scaler.update_initial(x)
    elif initial_state_procedure=='load':
        x = np.load('x.npy')
        scaler.update_initial(x.T)
    elif initial_state_procedure!='previous_iteration':
        policy_init = ray.get(network_id).policy_list(initial_state_procedure)
        x = ray.get(network_id).simulate_episode(np.zeros(obs_dim, 'int8'), policy_init)
        scaler.update_initial(x)
    else:
        run_policy(network_id, policy, scaler, logger, gamma, 0, skipping_steps, cycles_num=cycles_num,   episodes=1, time_steps=episode_duration)
    ###########################################################################





    iteration = 0  # count of policy iterations
    weights_set = []
    scaler_set = []
    while iteration < num_policy_iterations:
        # decrease clipping_range and learning rate
        iteration += 1
        alpha = 1. - iteration / num_policy_iterations
        policy.clipping_range = max(0.01, alpha*clipping_parameter)
        policy.lr_multiplier = max(0.05, alpha)

        print('Clipping range is ', policy.clipping_range)

        if iteration % 10 == 1:
            weights_set.append(policy.get_weights())
            scaler_set.append(copy.copy(scaler))

        trajectories = run_policy(network_id, policy, scaler, logger, gamma, iteration, skipping_steps, cycles_num,
                                      episodes=batch_size, time_steps=episode_duration) #simulation

        add_value(trajectories, val_func, scaler,
                  ray.get(network_id).next_state_list)  # add estimated values to episodes
        observes, disc_sum_rew_norm = add_disc_sum_rew(trajectories, policy, ray.get(network_id), gamma, lam, scaler, iteration)  # calculate values from data

        val_func.fit(observes, disc_sum_rew_norm, logger)  # update value function
        add_value(trajectories, val_func, scaler,
                  ray.get(network_id).next_state_list)  # add estimated values to episodes
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories, gamma, scaler)


        #scale, offset = scaler.get()
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, iteration)  # add various stats
        policy.update(observes, actions, np.squeeze(advantages), logger)  # update policy


        #print('V(0):', disc_sum_rew[0], val_func.predict([observes[0]])[0][0]/ scale[-1] + offset[-1])

        logger.write(display=True)  # write logger results to file and stdout

    weights = policy.get_weights()

    file_weights = os.path.join(logger.path_weights, 'weights_' + str(iteration) + '.npy')
    np.save(file_weights, weights)

    file_scaler = os.path.join(logger.path_weights, 'scaler_' + str(iteration) + '.npy')
    scale, offset = scaler.get()
    np.save(file_scaler, np.asarray([scale, offset]))
    weights_set.append(policy.get_weights())
    scaler_set.append(copy.copy(scaler))

    performance_evolution_all, ci_all = run_weights(network_id, weights_set, policy, scaler,cycles = 10**6)

    file_res = os.path.join(logger.path_weights, 'average_' + str(performance_evolution_all[-1]) + '+-' +str(ci_all[-1]) + '.txt')
    file = open(file_res, "w")
    for i in range(len(ci_all)):
        file.write(str(performance_evolution_all[i])+'\n')
    file.write('\n')
    for i in range(len(ci_all)):
        file.write(str(ci_all[i])+'\n')


    logger.close()
    policy.close_sess()
    val_func.close_sess()




if __name__ == "__main__":
    #A = [[-1, 1, 0], [0, -1, 0], [0, 0, -1]]
    #D = [[1, 0, 1], [0, 1, 0]]
    #alpha = [0.3, 0.0, 0.3]
    #mu = [2.0, 1., 2.0]
    # network = pn.ProcessingNetwork(A, D, alpha, mu, 'criss_cross')


    start_time = datetime.datetime.now()
    network = pn.ProcessingNetwork.from_name('criss_crossIH') # queuing network declaration
    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('time of queuing network object creation:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

    network_id = ray.put(network)


    parser = argparse.ArgumentParser(description=('Train policy for a queueing network '
                                                  'using Proximal Policy Optimizer'))

    parser.add_argument('-n', '--num_policy_iterations', type=int, help='Number of policy iterations to run',
                        default = 200)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor',
                        default = 1)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default = 1)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default = 0.003)
    parser.add_argument('-b', '--batch_size', type=int, help='Number of episodes per training batch',
                        default = 50)
    parser.add_argument('-m', '--hid1_mult', type=int, help='Size of first hidden layer for value and policy NNs',
                        default = 10)
    parser.add_argument('-t', '--episode_duration', type=int, help='Number of time-steps per an episode',
                        default = 10**6)
    parser.add_argument('-y', '--cycles_num', type=int, help='Number of cycles',
                        default = 5000)
    parser.add_argument('-c', '--clipping_parameter', type=float, help='Initial clipping parameter',
                        default = 0.2)
    parser.add_argument('-s', '--skipping_steps', type=int, help='Number of steps for which control is fixed',
                        default = 1)
    parser.add_argument('-i', '--initial_state_procedure', type=str,
                        help='procedure of generation intial states. Options: previous_iteration, LBFS, load, FBFS, cmu-policy, empty',
                        default = 'empty')


    args = parser.parse_args()
    main(network_id,  **vars(args))

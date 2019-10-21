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
from additional_functions import Logger, Scaler
import os
import argparse
import processing_network as pn
import random
import datetime

ray.init(temp_dir='/tmp/ray2', object_store_memory=10*10**9, redis_max_memory=10*10**9)

MAX_ACTORS = 25  # max number of parallel simulations

def diag_dot(A, B):
    # returns np.diag(np.dot(A, B))
    return np.einsum("ij,ji->i", A, B)


def run_policy(network_id, policy, scaler, logger, gamma, cycles_num,
               policy_iter_num, skipping_steps, episodes, time_steps):
    """
    Run given policy and collect data

    :param network_id: queuing network structure and first-order info
    :param policy: queuing network policy
    :param scaler: normalization values
    :param logger: metadata accumulator
    :param gamma: discount factor
    :param cycles_num: number of regenerative cycles in one trajectory
    :param policy_iter_num: policy iteration
    :param skipping_steps: number of steps when action does not change  ("frame-skipping" technique)
    :param episodes: number of parallel simulations (episodes)
    :param time_steps: max time steps in an episode
    :return: trajectories = (states, actions, rewards)

    """

    total_steps = 0
    action_optimal_sum = 0
    total_zero_steps = 0


    scale, offset = scaler.get()

    '''
    initial_states_set = random.sample(scaler.initial_states, k=episodes)
    trajectories, total_steps, action_optimal_sum, total_zero_steps = policy.run_episode(ray.get(network_id), scaler, time_steps, cycles_num, skipping_steps,  initial_states_set[0])
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
        trajectories.extend(accum_res[i][0])
        total_steps += accum_res[i][1]  # total time-steps
        action_optimal_sum += accum_res[i][2]  # absolute number of actions consistent with the "optimal policy"
        total_zero_steps += accum_res[i][3]  # absolute number of states for which all actions are optimal
    #################################################


    optimal_ratio = action_optimal_sum / (total_steps * skipping_steps)  # fraction of actions that are optimal
    # fraction of actions that are optimal excluding transitions when all actions are optimal
    pure_optimal_ratio = (action_optimal_sum - total_zero_steps)/ (total_steps * skipping_steps - total_zero_steps)

    ###### average performance computation #############
    cycle_lenght = 0
    reward_per_cycle = 0
    for t in trajectories:
        cycle_lenght+=len(t['rewards'])
        reward_per_cycle+=np.sum(t['rewards'])
    average_reward = reward_per_cycle / cycle_lenght
    ###################################################


    #### normalization of the states in data ####################
    if gamma < 1.0:
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]

    else:
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]
            z = t['rewards'] - average_reward
            t['rewards'] = z
    ##################################################################

    ####### data merging ###########################################
    unscaled_obs = np.concatenate([t['unscaled_obs'] for t in trajectories])
    observes = np.concatenate([t['observes'] for t in trajectories])
    rewards = np.concatenate([t['rewards'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])

    trajectory_whole = {
        'actions': actions,
        'rewards': rewards,
        'unscaled_obs': unscaled_obs,
        'observes': observes
    }
    ##########################################################



    ########## results report ##########################
    print('Average cost: ',  -average_reward)

    logger.log({'_AverageReward': -average_reward,
                'Steps': total_steps,
                'Zero steps':total_zero_steps,
                '% of optimal actions': int(optimal_ratio * 1000) / 10.,
                '% of pure optimal actions': int(pure_optimal_ratio * 1000) / 10.,
    })
    ####################################################
    return trajectory_whole

def add_disc_sum_rew(trajectory, policy, network, gamma, lam, scaler):
    """
    compute value function for further training of Value Neural Network
    :param trajectory: simulated data
    :param network: queuing network
    :param policy: current policy
    :param gamma: discount factor
    :param lam: lambda parameter in GAE
    :param scaler: normalization values
    """

    values = trajectory['values']
    observes = trajectory['observes']
    unscaled_obs = trajectory['unscaled_obs']


    ####### compute expectation of the value function of the next state ###########
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

    # expectation of the value function w.r.t the actual actions in data
    num_actions = len(network.actions)
    targets = trajectory['actions'][:, 0].reshape(-1)
    distr_fir = np.eye(num_actions)[targets]
    P_a = diag_dot(distr_fir, value_for_each_action)


    # td-error computing
    tds_pi = trajectory['rewards'] - values + P_pi[:, np.newaxis]#np.append(values[1:] * gamma, np.zeros((1, 1)), axis=0)#
    tds_a = trajectory['rewards'] - values + P_a[:, np.newaxis]#np.append(values[1:] * gamma, np.zeros((1, 1)), axis=0)#

    advantages = relarive_vf(unscaled_obs, td_pi = tds_pi, td_act = tds_a, lam=lam)# advantage function
    trajectory['advantages'] = np.asarray(advantages)

    # value function computing for futher neural network training
    #TODO: ensure that gamma<1 works
    if gamma < 1:
        disc_sum_rew = discount(trajectory['rewards'], gamma, trajectory['values'][-1])
    else:
        disc_sum_rew = relarive_vf(unscaled_obs, td_pi = tds_pi, td_act = tds_pi, lam=lam ) + values - values[0]
    trajectory['disc_sum_rew'] = disc_sum_rew
    scaler.update(np.hstack((trajectory['unscaled_obs'], trajectory['disc_sum_rew'])))

def discount(x, gamma, v_last):
        # return discounted value function
        disc_array = np.zeros(len(x))
        disc_array[-1] = v_last
        for i in range(len(x) - 2, -1, -1):
            disc_array[i] = x[i] + gamma * disc_array[i + 1]

        return disc_array

def relarive_vf(unscaled_obs, td_pi, td_act, lam):
    # return advantage function
    disc_array = np.copy(td_act)
    sum_tds = 0
    for i in range(len(td_pi) - 2, -1, -1):
        if np.sum(unscaled_obs[i+1]) != 0:
            sum_tds = td_pi[i+1] + lam * sum_tds
        else:
            sum_tds = 0
        disc_array[i] += sum_tds

    return disc_array



def add_value(trajectory_whole, val_func, scaler, possible_states):
    """
    # compute value function from the Value Neural Network
    :param trajectory_whole: simulated data
    :param val_func: Value Neural Network
    :param scaler: normalization values
    :param possible_states: transitions that are possible for the queuing network
    """

    scale, offset = scaler.get()
    # approximate value function for trajectory_whole['unscaled_obs']
    values = val_func.predict(trajectory_whole['observes'])
    trajectory_whole['values'] = values / scale[-1] + offset[-1]

    # approximate value function of the states where transitions are possible from trajectory_whole['unscaled_obs']
    values_set = np.zeros(( len(possible_states)+1, len(trajectory_whole['observes'])))
    values_set[-1] = np.squeeze(trajectory_whole['values'])
    for count, trans in enumerate(possible_states):
        new_obs =(trajectory_whole['unscaled_obs'] + trans - offset[:-1]) * scale[:-1]
        values = val_func.predict(new_obs)
        values = values / scale[-1] + offset[-1]
        values_set[count] = np.squeeze(values)

    trajectory_whole['values_set'] = values_set.T

def build_train_set(trajectory_whole, scaler):
    """
    # data pre-processing for training
    :param trajectory_whole:  simulated data
    :param scaler: normalization values
    :return: data for further Policy and Value neural networks training
    """

    scale, offset = scaler.get()
    unscaled_obs = trajectory_whole['unscaled_obs']
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    actions = trajectory_whole['actions']
    disc_sum_rew = trajectory_whole['disc_sum_rew']
    advantages = trajectory_whole['advantages']
    advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages



    ########## averaging value function estimations over all data ##########################
    states_sum = {}
    states_number = {}
    states_positions = {}

    for i in range(len(unscaled_obs)):
        if tuple(unscaled_obs[i]) not in states_sum:
            states_sum[tuple(unscaled_obs[i])] = disc_sum_rew[i]
            states_number[tuple(unscaled_obs[i])] = 1
            states_positions[tuple(unscaled_obs[i])] = [i]

        else:
            states_sum[tuple(unscaled_obs[i])] +=  disc_sum_rew[i]
            states_number[tuple(unscaled_obs[i])] += 1
            states_positions[tuple(unscaled_obs[i])].append(i)

    for key in states_sum:
        av = states_sum[key] / states_number[key]
        for i in states_positions[key]:
            disc_sum_rew[i] = av
    ########################################################################################

    return observes, unscaled_obs, actions, advantages, disc_sum_rew, states_sum, states_number, states_positions


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
        run_policy(network_id, policy, scaler, logger, gamma, cycles_num, 0, skipping_steps,   episodes=1, time_steps=episode_duration)
    ###########################################################################

    iteration = 0  # count of policy iterations
    while iteration < num_policy_iterations:
        # decrease clipping_range and learning rate
        iteration += 1
        alpha = 1. - iteration / num_policy_iterations
        if clipping_parameter == 0:
            policy.clipping_range = None
        else:
            policy.clipping_range = alpha*clipping_parameter
        policy.lr_multiplier = max(0.1, alpha)

        print('Clipping range is ', policy.clipping_range)
        trajectory_whole = run_policy(network_id, policy, scaler, logger, gamma, cycles_num, iteration, skipping_steps,
                                      episodes=batch_size, time_steps=episode_duration) #simulation

        add_value(trajectory_whole, val_func, scaler, ray.get(network_id).next_state_list)  # add estimated values to episodes
        add_disc_sum_rew(trajectory_whole, policy, ray.get(network_id), gamma, lam, scaler)  # calculate values from data
        observes, unscaled_obs, actions, advantages, disc_sum_rew, states_sum, states_number, states_positions = build_train_set(trajectory_whole, scaler)


        scale, offset = scaler.get()
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, iteration)  # add various stats
        policy.update(observes, actions, np.squeeze(advantages), logger)  # update policy


        disc_sum_rew_norm = (disc_sum_rew - offset[-1]) * scale[-1]
        val_func.fit(observes, disc_sum_rew_norm, logger)  # update value function

        logger.write(display=True)  # write logger results to file and stdout



    ############### run one long trajejectory to evaluate performance of the resulting policy ##############
    weights = policy.get_weights()
    file_weights = os.path.join(logger.path_weights, 'weights_'+str(num_policy_iterations)+'.npy')
    np.save(file_weights, weights) # save final Policy neural networks parameters to a file




    if scaler.initial_states_procedure == 'previous_iteration':
        initial_state_0 = np.zeros(obs_dim + 1)
    else:
        initial_state_0 = np.zeros(obs_dim)
    trajectories, total_steps, action_optimal_sum, zero_steps = policy.run_episode(ray.get(network_id), scaler,
                        time_steps=int(1./sum(ray.get(network_id).p_arriving) * 1000000), cycles_num=cycles_num*batch_size*10,
                        skipping_steps=1, initial_state = initial_state_0) # simulate a long trajectory




    cycle_lenght = 0
    reward_per_cycle = 0

    for t in trajectories:
        cycle_lenght+=len(t['rewards'])
        reward_per_cycle+=np.sum(t['rewards'])
    average_reward = -reward_per_cycle / cycle_lenght

    # save the performance evaluation results to the file
    file_res = os.path.join(logger.path_weights, 'average_'+str(average_reward)+' opt '+str(action_optimal_sum/ total_steps)+
                            'pure_opt '+str((action_optimal_sum-zero_steps)/ (total_steps-zero_steps))+'.npy')
    np.save(file_res, [average_reward, action_optimal_sum/ total_steps])
    print( 'average_'+str(average_reward)+' opt '+str(action_optimal_sum/ total_steps))


    ################################################




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
    network = pn.ProcessingNetwork.from_name('criss_crossIM') # queuing network declaration
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
                        default = 0.7)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default = 0.003)
    parser.add_argument('-b', '--batch_size', type=int, help='Number of episodes per training batch',
                        default = 25)
    parser.add_argument('-m', '--hid1_mult', type=int, help='Size of first hidden layer for value and policy NNs',
                        default = 10)
    parser.add_argument('-t', '--episode_duration', type=int, help='Number of time-steps per an episode',
                        default = 100*10**4)
    parser.add_argument('-y', '--cycles_num', type=int, help='Number of cycles',
                        default = 50)
    parser.add_argument('-c', '--clipping_parameter', type=float, help='Initial clipping parameter',
                        default = 0.2)
    parser.add_argument('-s', '--skipping_steps', type=int, help='Number of steps for which control is fixed',
                        default = 1)
    parser.add_argument('-i', '--initial_state_procedure', type=str,
                        help='procedure of generation intial states. Options: previous_iteration, LBFS, load, FBFS, cmu-policy, empty',
                        default = 'empty')


    args = parser.parse_args()
    main(network_id,  **vars(args))

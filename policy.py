import numpy as np
import tensorflow as tf
import ray.experimental
import datetime


class Policy(object):
    """ Policy Neural Network """

    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult,  clipping_range=0.2, temp=2.0):
        """
        :param obs_dim: num observation dimensions
        :param act_dim: num action dimensions
        :param kl_targ: target KL divergence between pi_old and pi_new
        :param hid1_mult: size of first hidden layer, multiplier of obs_dim
        :param clipping_range:
        :param temp: temperature parameter
        """
        self.temp = temp
        self.beta = 3  # dynamically adjusted D_KL loss multiplier
        self.kl_targ = kl_targ
        self.hid1_mult = hid1_mult
        self.epochs = 5
        self.lr = None
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clipping_range = clipping_range

        self._build_graph()
        #self._init_session()



    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()
            self.sess = tf.Session(graph=self.g)
            self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
            self.sess.run(self.init)


    def _placeholders(self):
        """ Define placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')

        self.act_ph = tf.placeholder(tf.int32, (None, len(self.act_dim)), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        self.lr_ph = tf.placeholder(tf.float32, (), 'eta') # learning rate:
        self.old_act_prob_ph = []
        for i in range(len(self.act_dim)):# split the output layer over queuing stations
            self.old_act_prob_ph.append(tf.placeholder(tf.float32, (None, self.act_dim[i]), 'old_act_prob'+str(i)))



    def _policy_nn(self):
        """ Neural Network architecture for policy approximation function
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        hid3_size = len(self.act_dim) * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 5. * 10**(-4)  # 9e-4 empirically determined
        # 3 hidden layers with tanh activations
        out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                 stddev=np.sqrt(1 / hid1_size)), name="h2")

        out_last = []
        act_prob_out = []
        for i in range(len(self.act_dim)):
            out_last.append(tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3"+str(i)))

            act_prob_out.append( tf.layers.dense(tf.divide(out_last[i], self.temp), self.act_dim[i], tf.nn.softmax,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / hid3_size)), name="act_prob"+str(i)))

        self.act_prob_out = [act_prob_out[i] for i in range(len(act_prob_out))]

    def _logprob(self):
        """
        Calculate probabilities using previous step's model parameters and new parameters being trained.
        """
        act_probs = []
        act_probs_old = []
        for i in range(len(self.act_dim)):
            act_probs.append(self.act_prob_out[i])
            act_probs_old.append(self.old_act_prob_ph[i])
        for i in range(len(self.act_dim)):
            # probabilities of actions which agent took with policy
            act_probs[i] = tf.reduce_sum(act_probs[i] * tf.one_hot(indices=self.act_ph[:,i], depth=act_probs[i].shape[1]), axis=1)


            # probabilities of actions which agent took with old policy
            act_probs_old[i] = tf.reduce_sum(act_probs_old[i] * tf.one_hot(indices=self.act_ph[:,i], depth=act_probs_old[i].shape[1]), axis=1)



        self.act_probs_old = np.prod(act_probs_old, axis=0)
        self.act_probs = np.prod(act_probs, axis = 0)


    def _kl_entropy(self):
        """
        Calculate KL-divergence between old and new distributions
        """

        self.entropy = 0
        self.kl = 0
        for i in range(len(self.act_dim)):
            kl = tf.reduce_sum(self.act_prob_out[i] * (tf.log(tf.clip_by_value(self.act_prob_out[i], 1e-10, 1.0))
                                                       - tf.log(tf.clip_by_value(self.old_act_prob_ph[i], 1e-10, 1.0))), axis=1)
            entropy = tf.reduce_sum(self.act_prob_out[i] * tf.log(tf.clip_by_value(self.act_prob_out[i], 1e-10, 1.0)), axis=1)
            self.entropy += -tf.reduce_mean(entropy, axis=0)# sum of entropy of pi(obs)
            self.kl += tf.reduce_mean(kl, axis=0)


    def _loss_train_op(self):
        """
        Calculate the PPO loss function
        """

        ratios = tf.exp(tf.log(self.act_probs) - tf.log(self.act_probs_old))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clipping_range, clip_value_max=1 + self.clipping_range)
        loss_clip = tf.minimum(tf.multiply(self.advantages_ph, ratios), tf.multiply(self.advantages_ph, clipped_ratios))

        self.loss = -tf.reduce_mean(loss_clip)#- self.entropy*0.0001
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)


    def sample(self, obs, stochastic=True):
        """
        :param obs: state
        :return: if stochastic=True returns pi(a|x), else returns distribution with prob=1 on argmax[ pi(a|x) ]
        """

        feed_dict = {self.obs_ph: obs}
        if stochastic:
            return self.sess.run(self.act_prob_out, feed_dict=feed_dict)
        else:
            determ_prob = []
            for i in range(len(self.act_dim)):
                pr = self.sess.run(self.act_prob_out[i], feed_dict=feed_dict)
                inx = np.argmax(pr)
                ar = np.zeros(self.act_dim[i])
                ar[inx] = 1
                determ_prob.extend([ar[np.newaxis]])
            return determ_prob


    def update(self, observes, actions, advantages, logger):
        """
        Policy Neural Network update
        :param observes: states
        :param actions: actions
        :param advantages: estimation of antantage function at observed states
        :param logger: statistics accumulator
        """

        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.lr_ph: self.lr * self.lr_multiplier}

        old_act_prob_np = self.sess.run(self.act_prob_out, feed_dict)  # actions probabilities w.r.t the current policy
        for i in range(len(self.act_dim)):
            feed_dict[self.old_act_prob_ph[i]] = old_act_prob_np[i]

        loss = 0
        kl = 0
        entropy = 0
        for e in range(self.epochs): # training
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                print('early stopping: D_KL diverges badly')
                break

        # actions probabilities w.r.t the new and old (current) policies
        act_probs, act_probs_old = self.sess.run([self.act_probs, self.act_probs_old], feed_dict)
        ratios = np.exp(np.log(act_probs) - np.log(act_probs_old))
        if self.clipping_range is not None:
            clipping_range = self.clipping_range
        else:
            clipping_range = 0

        logger.log({'PolicyLoss': loss,
                    'Clipping' : clipping_range,
                    'Max ratio': max(ratios),
                    'Min ratio': min(ratios),
                    'Mean ratio': np.mean(ratios),
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier })





    def run_episode(self, network, scaler, time_steps, cycles_num,  skipping_steps, initial_state):
        """
        One episode simulation
        :param network: queuing network
        :param scaler: normalization values
        :param time_steps: max number of time steps
        :param cycles_num: number of regenerative cycles to simulate
        :param skipping_steps: number of steps for which control is fixed
        :param initial_state: initial state for the episode
        :return: collected data
        """

        trajectories = [] # collect regenerative cycles
        policy_buffer = {} # save action disctribution of visited states

        total_steps = 0 # count steps
        action_optimal_sum = 0 # count actions that coinside with the optimal policy
        total_zero_steps = 0 # count states for which all actions are optimal

        observes = np.zeros((time_steps, network.buffers_num))
        actions = np.zeros((time_steps, network.stations_num), 'int8')
        rewards = np.zeros((time_steps, 1))
        unscaled_obs = np.zeros((time_steps, network.buffers_num), 'int32')


        for cyc in range(cycles_num):
            scale, offset = scaler.get()

            ##### modify initial state according to the method of intial states generation######
            if scaler.initial_states_procedure =='previous_iteration':
                if sum(initial_state[:-1]) > 300 :
                    initial_state = np.zeros(network.buffersNum+1, 'int8')
                state = np.asarray(initial_state[:-1],'int32')
            else:
                state = np.asarray(initial_state, 'int32')

            ###############################################################

            t = 0
            while t < 1 or np.sum(state)!=0: # run until visit to the empty state (regenerative state)
                unscaled_obs[t] = state
                state_input = (state - offset[:-1]) * scale[:-1]  # center and scale observations

                ###### compute action distribution according to Policy Neural Network for state###
                if tuple(state) not in policy_buffer:
                    act_distr = self.sample([state_input])
                    policy_buffer[tuple(state)] =act_distr
                distr = policy_buffer[tuple(state)][0][0] # distribution for each station
                for ar_i in range(1, network.stations_num):
                    distr = [a*b for a in distr for b in policy_buffer[tuple(state)][ar_i][0]]
                distr = distr / sum(distr)
                ############################################

                """
                act_ind - all possible actions are numerated GLOBALLY as 0, 1, 2, ...
                action_full - buffers that have priority are equal to 1, otherwise 0
                action_for_server - all possible actions FOR EACH STATIONS are numerated as 0, 1, 2, ...
                
                For the simple reentrant line. 
                If priority is given to the first class:
                    act_ind = [0]
                    action_full = [1, 1, 0]
                    action_for_server = [0, 0]
                    
                If priority is given to the third class:
                    act_ind = [1]
                    action_full = [0, 1, 1]
                    action_for_server = [1, 0]
                """
                act_ind = np.random.choice(len(distr), 1, p=distr) # sample action according to distribution 'distr'
                action_full = network.dict_absolute_to_binary_action[act_ind[0]]
                action_for_server = network.dict_absolute_to_per_server_action[act_ind[0]]

                ######### check optimality of the sampled action ################
                if state[0]<140 and state[1]<140 and state[2]<140:
                      if state[0]==0 or state[2]==0:
                          total_zero_steps += 1

                      action_optimal = network.comparison_policy[tuple(state)]
                      if all(action_full == action_optimal) or state[0]==0 or state[2]==0:
                          action_optimal_sum += 1
                #######################



                rewards[t] = np.asarray(-np.sum(state))
                state = network.next_state(state, action_full)
                actions[t] = action_for_server
                observes[t] = state_input


                for i in range(skipping_steps-1):
                     rewards[t] += -np.sum(state)
                     state = network.next_state(state, action_full) # move to the next state
                t+=1

            total_steps += len(actions[:t])
            # record simulation
            trajectory = {#'observes': observes,
                          'actions': actions[:t],
                          'rewards': rewards[:t] / skipping_steps,
                          'unscaled_obs': unscaled_obs[:t]
                      }

            trajectories.append(trajectory)
        return trajectories, total_steps, action_optimal_sum, total_zero_steps


    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def get_obs_dim(self):
        return self.obs_dim

    def get_hid1_mult(self):
        return self.hid1_mult

    def get_act_dim(self):
        return self.act_dim

    def get_weights(self):
        return self.variables.get_weights()

    def set_weights(self, weights):
        # Set the weights in the network.
        self.variables.set_weights(weights)

    def get_kl_targ(self):
        return self. kl_targ
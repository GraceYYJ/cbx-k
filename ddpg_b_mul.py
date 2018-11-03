# encoding: utf-8
import numpy as np
import math
import my_common as mc

import torch
from quantization_env import QuantizationEnv
from ddpg import DDPG
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition
import torch.multiprocessing as mp
from queue import Queue

'''
def train(model):
    # Construct data_loader, optimizer, etc.
    for data, labels in data_loader:
        optimizer.zero_grad()
        loss_fn(model(data), labels).backward()
        optimizer.step()  # This will update the shared parameters

if __name__ == '__main__':
    num_processes = 4
    model = MyModel()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
      p.join()
'''

class DDPGB(object):
    #x是标准X的转置矩阵
    #b是标准二进制b矩阵
    #c是标准codebook矩阵
    def __init__(self, c, x, b, action_output_num, cr, replay_size=1000000, ou_noise=True, param_noise=False, noise_scale=0.3, final_noise_scale=0.3):
        self.C = c
        self.X = x
        self.B = b
        self.intB = None
        #self.new_B = []
        self.b_q = Queue()
        #self.env = None
        #self.agent = None
        self.replay_size = replay_size
        #self.memory = ReplayMemory(replay_size)
        self.ou_noise = ou_noise
        self.cr = cr
        self.iter_code = [1, 2, 3, 3, 5, 5, 5, 7, 7, 7, 9, 9, 9, 12, 12, 12, 12, 15]
        self.noise_scale = noise_scale
        self.final_noise_scale = final_noise_scale
        self.ounoise = OUNoise(action_output_num) if ou_noise else None
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=noise_scale, adaptation_coefficient=1.05) if param_noise else None

    def update_B(self, c, b, cr):
        self.C = c
        self.B = b
        self.cr = cr
        #self.new_B = []
        self.intB = None

    #mul_i用于在存入数据时，记录当前数据属于哪一个i，方便还原后续的X顺序
    def subset_B(self, mul_i, sub_X, sub_B, hash_d, coff, gamma, tau, hidden_size=128, data_width=8, spp_num_outputs=[4, 2, 1], num_episodes=10, exploration_end=150, batch_size=128, updates_per_step=5):
        for i_X in range(sub_X.shape[0]):
            x_rewards = []
            new_B = []
            x_total_numsteps = 0
            x_updates = 0
            best_bb = -1000000
            best_action = None
            memory = ReplayMemory(self.replay_size)

            env = QuantizationEnv(self.C, sub_X[i_X], sub_B[i_X], hash_d, coff)
            agent = DDPG(gamma, tau, hidden_size, env.action, hash_d, data_width, spp_num_outputs)

            for i_episode in range(self.iter_code[self.cr] * num_episodes):
                state = torch.Tensor([env.reset()])
                if self.ou_noise:
                    self.ounoise.scale = (self.noise_scale - self.final_noise_scale) * max(0, exploration_end - i_episode) // exploration_end + self.final_noise_scale
                    self.ounoise.reset()
                if self.param_noise:
                    agent.perturb_actor_parameters(self.param_noise)
                episode_reward = 0
                train_max_iter = 100 * self.iter_code[self.cr]
                train_cur_iter = 0
                continue_right_reward = 0
                while True:
                    train_cur_iter = train_cur_iter + 1
                    action = agent.select_action(state, self.ounoise, self.param_noise)
                    next_state, reward, done, bb = env.step(action)
                    if bb > best_bb:
                        best_bb = bb
                        best_action = action
                    x_total_numsteps += 1
                    episode_reward += reward
                    print("best_b:", best_bb)
                    print("current_action:", action)
                    action = torch.Tensor(action.cpu()).cuda()
                    mask = torch.Tensor([not done])
                    next_state = torch.Tensor([next_state])
                    reward = torch.Tensor([reward])

                    if reward > 0:
                        continue_right_reward = continue_right_reward + 1
                        memory.push(state, action, mask, next_state, reward)
                    else:
                        continue_right_reward = 0

                    if continue_right_reward > train_max_iter // 2:
                        done = True
                    state = next_state

                    if len(memory) > batch_size:
                        for _ in range(self.iter_code[self.cr] * updates_per_step):
                            transitions = memory.sample(batch_size)
                            batch = Transition(*zip(*transitions))
                            value_loss, policy_loss = agent.update_parameters(batch)

                            print("gross:", self.cr, "epoch:", i_episode, " i element:", i_X, "updates", x_updates,
                                  "value_loss:", value_loss, " policy_loss:", policy_loss)
                            if value_loss < 0.1:
                                done = True

                            x_updates += 1
                    if done:
                        break

                    if self.param_noise:
                        episode_transitions = memory.memory[memory.position - batch_size:memory.position]
                        states = torch.cat([transition[0] for transition in episode_transitions], 0)
                        unperturbed_actions = agent.select_action(states, None, None)
                        perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

                        ddpg_dist = ddpg_distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
                        self.param_noise.adapt(ddpg_dist)

                    if train_max_iter < train_cur_iter:
                        break

                    x_rewards.append(episode_reward)
                if i_episode % 10 == 0 and i_episode != 0:
                    state = torch.Tensor([env.reset()])
                    episode_reward = 0
                    test_max_iter = 15 * self.iter_code[self.cr]
                    test_cur_iter = 0
                    while True:
                        test_cur_iter = test_cur_iter + 1
                        action = agent.select_action(state)
                        print("test_action:", action)
                        next_state, reward, done, bb = env.step(action)
                        if bb > best_bb:
                            best_bb = bb
                            best_action = action
                        print("test_best_b:", best_bb)
                        print("test_state:", state)
                        episode_reward += reward
                        next_state = torch.Tensor([next_state])
                        state = next_state
                        if done:
                            break
                        if test_cur_iter > test_max_iter:
                            break

                        x_rewards.append(episode_reward)

                # print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, x_total_numsteps, x_rewards[-1], np.mean(x_rewards[-10:])))
            new_B = np.hstack((new_B, mc.trans_b10_vector2(best_action.cpu().numpy(), hash_d)))
        #return np.array(self.new_B).reshape(self.B.shape[1], self.B.shape[0]).T
        #此处应当将new_B和多线程标识mul_i形成一个元组存储公共队列self.b_q中，在上层函数程序中取出拼接

    #hash_d是每个输出对应的二进制维度
    # mul_num代表要开启的进程数量
    # coff代表reward中的权重比例[0.2, 0.8]
    def generate_B(self, mul_num, hash_d, coff, gamma, tau, hidden_size=128, data_width=8, spp_num_outputs=[4, 2, 1], num_episodes=10, exploration_end=150, batch_size=128, updates_per_step=5):
        self.intB = mc.trans_binmat_intmat(self.B, hash_d)
        processes = []
        step = self.X.shape[0] // mul_num
        for rank in range(mul_num):
            p = mp.Process(target=self.subset_B, args=(rank, self.X[rank * step:(rank + 1) * step], self.intB[rank * step:(rank + 1) * step], hash_d, coff, gamma, tau, hidden_size, data_width, spp_num_outputs, num_episodes, exploration_end, batch_size, updates_per_step, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        #此处，从self.b_q中将数据取出，按mul_i顺序拼接new_B数据(使用np.hstack后reshape),形成B矩阵，return给主程序







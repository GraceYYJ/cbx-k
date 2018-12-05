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


class DDPGB(object):
    # x是数值向量
    # b是码值向量
    # c是标准codebook矩阵
    # action_output_num是码值输出的维度
    # replay_size是meomery队列的最大长度
    # new_b代表新计算得到的b
    # env代表action和obersevation的产生环境
    # agent代表实际的ddpg执行体
    # 保留这些噪声参数只是为了能够进入到需要随机探索的部分
    def __init__(self, C, b, x, action_output_num, actor_size, replay_size=1000000, ou_noise=True, param_noise=True,
                 noise_scale=0.3, final_noise_scale=0.3):
        self.C = C
        self.b = b
        self.x = x
        self.hd = action_output_num
        self.actor_size = actor_size
        self.memory = ReplayMemory(replay_size)
        self.new_b = None
        self.env = None
        self.agent = None
        self.ou_noise = ou_noise
        self.noise_scale = noise_scale
        self.final_noise_scale = final_noise_scale
        self.ounoise = OUNoise(action_output_num) if ou_noise else None
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=noise_scale,
                                                  adaptation_coefficient=1.05) if param_noise else None

    def update_B(self, c, b, x):
        self.C = c
        self.b = b
        self.x = x

    # 备选coff代表reward中的权重比例[0.2, 0.8]
    def generate_B(self, coff, gamma, tau, hidden_size, num_inputs, actor_size, num_episodes=60000,
                   exploration_end=150, batch_size=512, updates_per_step=5000):

        self.env = QuantizationEnv(self.C, self.b, self.x, self.hd, coff)
        self.agent = DDPG(gamma, tau, hidden_size, self.env.action_bin, num_inputs, actor_size)
        rewards = []
        total_numsteps = 0
        updates = 0
        max_trail = 10000
        best_bb = 10000

        # 开启num_episodes次最佳方案寻找
        for i_episode in range(num_episodes):
            state = torch.Tensor([self.env.reset()])
            if self.ou_noise:
                self.ounoise.scale = (self.noise_scale - self.final_noise_scale) * max(0, exploration_end - i_episode) \
                                     / exploration_end + self.final_noise_scale
                self.ounoise.reset()
            if self.param_noise:
                self.agent.perturb_actor_parameters(self.param_noise)

            episode_reward = 0
            continuous_neg = 0
            continuous_pos = 0
            temp_trail = 0

            control_bit = 0
            next_state = self.env.compute_Cbx(self.b)
            next_state = torch.Tensor([next_state])

            while True:
                # yyj
                if control_bit > 15:
                    control_bit = control_bit % 16
                state = next_state
                action = self.agent.select_action(state, self.ounoise, self.param_noise)
                next_state, reward, done, bb = self.env.step(action, control_bit,self.actor_size)
                # print(control_bit, next_state[0], reward, done, bb)
                control_bit = control_bit + 1
                total_numsteps += 1
                episode_reward += reward
                # bb是c_v值
                if best_bb > bb:
                    best_bb = bb
                    self.new_b = action

                if reward > 0:
                    continuous_pos += 1
                    continuous_neg = 0
                    if continuous_pos > 10:
                        done = True
                if reward < 0:
                    continuous_neg += 1
                    continuous_pos = 0
                    if continuous_neg > 10:
                        done = True
                if temp_trail > max_trail:
                    done = True

                action = torch.Tensor(action)
                mask = torch.Tensor([not done])
                next_state = torch.Tensor([next_state])
                reward = torch.Tensor([reward])

                self.memory.push(state, action, mask, next_state, reward)

                # state = next_state
                temp_trail += 1

                # memorysize还不够，不会进入
                if len(self.memory) > batch_size:
                    for _ in range(updates_per_step):
                        transitions = self.memory.sample(1)
                        batch = Transition(*zip(*transitions))

                        # value_loss属于右边的网络，policy_loss属于左边的网络
                        value_loss, policy_loss = self.agent.update_parameters(batch)
                        print("epoch:", i_episode, "updates", updates, "value_loss:",
                              value_loss, " policy_loss:", policy_loss)

                        updates += 1
                if done:
                    break

            if self.param_noise:
                episode_transitions = self.memory.memory[self.memory.position - batch_size:self.memory.position]
                states = torch.cat([transition[0] for transition in episode_transitions], 0)
                unperturbed_actions = self.agent.select_action(states, None, None)
                perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

                ddpg_dist = ddpg_distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
                self.param_noise.adapt(ddpg_dist)

            rewards.append(episode_reward)
            continuous_neg = 0
            continuous_pos = 0
            temp_trail = 0
            if i_episode % 10 == 0 and i_episode != 0:
                state = torch.Tensor([self.env.reset()])
                episode_reward = 0
                control_bit = 0
                while True:
                    action = self.agent.select_action(state)
                    next_state, reward, done, bb = self.env.step(action.numpy()[0], control_bit)
                    episode_reward += reward
                    if best_bb > bb:
                        best_bb = bb
                        self.new_b = action

                    if reward > 0:
                        continuous_pos += 1
                        continuous_neg = 0
                        if continuous_pos > 10:
                            done = True
                    if reward < 0:
                        continuous_neg += 1
                        continuous_pos = 0
                        if continuous_neg > 10:
                            done = True
                    if temp_trail > max_trail:
                        done = True

                    next_state = torch.Tensor([next_state])

                    state = next_state
                    temp_trail += 1

                    if done:
                        break

                rewards.append(episode_reward)
                print(
                    "Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps,
                                                                                             rewards[-1],
                                                                                             np.mean(rewards[-10:])))

        return self.new_b

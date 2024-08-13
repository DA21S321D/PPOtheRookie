from network import FeedForwardNN
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
from torch import nn
import numpy as np

class PPO:
    def __init__(self,env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        #初始化网络
        self.actor = FeedForwardNN(self.obs_dim,self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim,1)#根据当前状态评分，所以输入是obs，输出是1维度
        self._init_hyperparameters()

        #get_aciton使用的内容：
        #协方差值
        self.cov_var = torch.full(size= (self.act_dim,),fill_value = 0.5)
        #协方差矩阵,使用协方差值构成的对角协方差矩阵
        self.cov_mat = torch.diag(self.cov_var)

        #计算reward to go用到的
        self.gamma = 0.95

        self.n_updates_per_iteration = 5
        self.clip = 0.2


        self.lr = 0.005
        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)

        self.critic_optim = Adam(self.critic.parameters(),lr = self.lr)

    def learn(self,total_timesteps):


        t_so_far = 0 #time steps simulated so far
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            #计算V评分
            V, _= self.evaluate(batch_obs, batch_acts)
            # 计算优势
            A_k = batch_rtgs - V.detach()
            #标准化优势（作者使用的的技巧）
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip , 1 + self.clip) * A_k # torch.clamp它将参数 1 或参数 2 和参数 3 之间的比率绑定为各自的下限和上限。
                actor_loss = (-torch.min(surr1,surr2)).mean()

                #反向传播更新actor
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                critic_loss = nn.MSELoss()(V, batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            # 计算我们收集了这个批次的多少个时间步
            t_so_far += np.sum(batch_lens)
    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episodes = 1600

    def rollout(self):

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = [] #rewars to go
        batch_lens = [] #episodic length in batch






        t = 0 #number of timesteps of this batch run so far
        while t<self.timesteps_per_batch:
            #当前episode的奖励列表
            ep_rews =[]
            #初始化环境
            obs = self.env.reset()
            obs = obs[0]#去除返回空的那部分
            done = False
            #一个episido：
            for ep_t in range(self.max_timesteps_per_episodes):
                t+=1
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)
                #收集奖励，动作，和动作概率
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)#记录这个episode所进行的时间长度（中途完成任务长度就不是最大值）
            batch_rews.append(ep_rews)

        #需要将batch记录的数据转给换成张量，并计算reward to go
        batch_obs = torch.tensor(batch_obs, dtype = torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)#batch_rews使用的是append，每个episode的reward都是列表中的一个元素

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    def get_action(self,obs):

        #actor先通过网络获取一个动作，当作“均值”
        mean = self.actor(obs)

        #根据这个“均值”和协方差矩阵创建一个分布
        dist = MultivariateNormal(mean, self.cov_mat)

        #从这个分布中取动作值和动作概率
        action = dist.sample()
        log_prob = dist.log_prob(action)


        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self,batch_rews):

        batch_rtgs = []

        for ep_rews in reversed(batch_rews):#反方向计算奖励

            discounted_reward = 0#迄今为止的折扣奖励
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma

                batch_rtgs.insert(0,discounted_reward)
            #转换成张量
        batch_rtgs = torch.tensor(batch_rtgs, dtype = torch.float)

        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze() #每个episide的obs界限去掉了，一个batch所有的V评分在一起
        #计算优势概率
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs


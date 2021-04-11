import numpy as np
import scipy.special
#import click
import gym
batch_size=2000
discount=0.99
n_itrs=200
render=False
use_baseline=True
env_id='CartPole-v0'
natural_step_size=0.01
env = gym.make('CartPole-v0')
nprs=np.random.RandomState
rng = np.random.RandomState(42)
env = gym.make('CartPole-v0')
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
def weighted_sample(logits, rng=np.random):
    weights = softmax(logits)
    return min(
        int(np.sum(rng.uniform() > np.cumsum(weights))),
        len(weights) - 1
    )
def compute_entropy(logits):
    """
    params: A matrix of size N * |A|
    return: A vector of size N
    """
    logp = log_softmax(logits)
    return -np.sum(logp * np.exp(logp), axis=-1)
def numerical_grad(f, x, eps=1e-8):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xplus = np.array(x)
        xplus[i] += eps
        fplus = f(xplus)
        xminus = np.array(x)
        xminus[i] -= eps
        fminus = f(xminus)
        grad[i] = (fplus - fminus) / (2 * eps)
    return grad
def log_softmax(logits):
    return logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
def softmax(logits):
    x = logits
    x = x - np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=-1, keepdims=True)


def include_bias(x):
    return np.concatenate([x, np.ones_like(x[..., :1])], axis=-1)

def compute_logits(theta, ob):
    """
    theta: A matrix of size |A| * (|S|+1)
    ob: A vector of size |S|
    return: A vector of size |A|
    """
    #ob_1 = include_bias(ob)
    logits = ob.dot(theta.T)
    return logits


def get_logp_action(theta, ob, action):
    """
     theta: A matrix of size |A| * (|S|+1)
     ob: A vector of size |S|
     action: An integer
     return: A scalar
    """
    return log_softmax(compute_logits(theta, ob))[action]


def get_grad_logp_action(theta, ob, action):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :param action: An integer
    :return: A matrix of size |A| * (|S|+1)
    """
    e_a = np.zeros(theta.shape[0]) # |A|
    e_a[action] = 1.
    #ob_1= include_bias(ob) # |S| + 1
    logits = np.dot(ob,theta.T) # |S| + 1  * (|S|+1) * |A|
    return np.outer(e_a - softmax(logits), ob)  # (|A| - |A|) * |S| + 1


def get_action(theta, ob, rng=np.random):
    """
    theta: A matrix of size |A| * (|S|+1)
    ob: A vector of size |S|
    return: An integer
    """
    return weighted_sample(compute_logits(theta, ob), rng=rng)
def compute_fisher_matrix(theta, get_grad_logp_action, all_observations, all_actions):
    """
    theta: A matrix of size |A| * (|S|+1)
    get_grad_logp_action: A function, mapping from (theta, ob, action) to the gradient (a matrix 
    of size |A| * (|S|+1) )
    all_observations: A list of vectors of size |S|
    all_actions: A list of vectors of size |A|
    return: A matrix of size (|A|*(|S|+1)) * (|A|*(|S|+1)), i.e. #columns and #rows are the number of 
    entries in theta
    """
    d = len(theta.flatten()) 
    F = np.zeros((d, d)) # Shape = (|A|* (|S|+1), |A|* (|S|+1))
    for ob, action in zip(all_observations, all_actions):
        grad_logp = get_grad_logp_action(theta, ob, action).flatten() # Shape = (|A|* (|S|+1),)
        F += np.outer(grad_logp, grad_logp)
        F /= len(all_actions)
    return F

def compute_natural_gradient(F, grad, reg=1e-4):
    F_inv = np.linalg.inv(F + reg * np.eye(F.shape[0]))
    natural_grad = F_inv.dot(grad.flatten()) # ∇^{~}J= F^(-1) . ∇ J
    return np.reshape(natural_grad, grad.shape)
#Adaptive Step Size
def compute_step_size(F, natural_grad, natural_step_size):
    g_nat = natural_grad.flatten()
    #if g_nat.T.dot(F.dot(g_nat)) != np.zeros(g_nat.T.dot(F.dot(g_nat)).shape):
    return np.sqrt((2 * natural_step_size) / (g_nat.T.dot(F.dot(g_nat))))
  

env.seed(42)
#timestep_limit = env.spec.timestep_limit
timestep_limit=2000
baselines = np.zeros(timestep_limit)
theta = np.random.normal(scale=0.1, size=(action_dim, obs_dim ))

for itr in range(n_itrs):
        # Collect trajectory loop
        n_samples = 0
        grad = np.zeros_like(theta)
        episode_rewards = []

        # Store cumulative returns for each time step
        all_returns = [[] for _ in range(timestep_limit)]

        all_observations = []
        all_actions = []

        while n_samples < batch_size:
            observations = []
            actions = []
            rewards = []
            ob = env.reset()
            done = False
            # Only render the first trajectory
            render_episode = n_samples == 0
            # Collect a new trajectory
            while not done:
                action = get_action(theta, ob, rng)
                next_ob, rew, done, _ = env.step(action)
                observations.append(ob)
                actions.append(action)
                rewards.append(rew)
                ob = next_ob
                n_samples += 1
                #if render and render_episode:
                env.render()
            # Go back in time to compute returns and accumulate gradient
            # Compute the gradient along this trajectory
            R = 0.
            for t in reversed(range(len(observations))):
                def compute_update(discount, R_tplus1, theta, s_t, a_t, r_t, b_t, get_grad_logp_action):
                    """
                    discount: A scalar
                    R_tplus1: A scalar
                    theta: A matrix of size |A| * (|S|+1)
                    s_t: A vector of size |S|
                    a_t: Either a vector of size |A| or an integer, depending on the environment
                    r_t: A scalar
                    b_t: A scalar
                    get_grad_logp_action: A function, mapping from (theta, ob, action) to the gradient (a 
                    matrix of size |A| * (|S|+1) )
                    return: A tuple, consisting of a scalar and a matrix of size |A| * (|S|+1)
                    """
                    R_t = (discount * R_tplus1) + r_t
                    pg_theta = get_grad_logp_action(theta, s_t, a_t) * (R_t -b_t)
                    return R_t, pg_theta

                
               

                R, grad_t = compute_update(
                    discount=discount,
                    R_tplus1=R,
                    theta=theta,
                    s_t=observations[t],
                    a_t=actions[t],
                    r_t=rewards[t],
                    b_t=baselines[t],
                    get_grad_logp_action=get_grad_logp_action
                )
                all_returns[t].append(R)
                grad += grad_t

            episode_rewards.append(np.sum(rewards))
            all_observations.extend(observations)
            all_actions.extend(actions)

        def compute_baselines(all_returns):
            """
            all_returns: A list of size T, where the t-th entry is a list of numbers, denoting the returns 
            collected at time step t across different episodes
            return: A vector of size T
            """
            baselines = np.zeros(len(all_returns))
            for t in range(len(all_returns)):
                if len(all_returns[t]) == 0:
                    baselines[t] = 0
                else:
                    baselines[t] = np.mean(all_returns[t])
            return baselines

        if use_baseline:
            baselines = compute_baselines(all_returns)
        else:
            baselines = np.zeros(timestep_limit)

        # Roughly normalize the gradient
        grad = grad / (np.linalg.norm(grad) + 1e-8)
        #Compute Fisher matrix
        F = compute_fisher_matrix(theta=theta, get_grad_logp_action=get_grad_logp_action,
                                      all_observations=all_observations, all_actions=all_actions)
        natural_grad = compute_natural_gradient(F, grad)
        #step_size = 0.1*compute_step_size(F, natural_grad, natural_step_size)
        #Update step
        theta += 0.001  * natural_grad
        logits = compute_logits(theta, np.array(all_observations))
        ent = np.mean(compute_entropy(logits))
        perp = np.exp(ent)

        print((itr, np.mean(episode_rewards), ent, perp, np.linalg.norm(theta)))
if __name__ == "__main__":
    pass

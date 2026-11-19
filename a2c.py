"""
A2C (Advantage Actor-Critic) — minimalRL 스타일 단일 파일 구현

논문: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016 (A3C)
     A2C는 A3C의 "동기(synchronous)" 버전이다. (OpenAI Baselines에서 정리됨)
     비동기로 여러 워커가 각자 gradient를 보내는 대신, N개의 환경을 동시에 한 스텝씩 진행시키고
     모은 배치로 한 번에 업데이트한다. GPU 활용이 좋고 성능은 A3C와 비슷하거나 더 낫다.

[알고리즘 핵심 아이디어]
  Actor  π(a|s; θ) : 정책. Policy gradient로 학습
       ∇θ J ≈ E[ ∇θ log π(a_t|s_t) · A(s_t, a_t) ]
  Critic V(s; θ_v) : 상태 가치. Advantage A = R_t - V(s_t)를 계산하기 위한 baseline
  → baseline을 빼면 기댓값은 그대로이면서 분산이 크게 줄어든다.

[논문에서 성능/안정성을 위해 도입한 도구 → 이 파일의 함수]
  1) 병렬 actor (여러 환경 동시 실행) .... SyncVecEnv  (리플레이 버퍼 없이 데이터 상관관계 제거)
  2) n-step return ..................... compute_nstep_returns()  (편향-분산 절충, 보상 전파 가속)
  3) Advantage (baseline 차감) ......... compute_a2c_loss() 안의 advantage
  4) Entropy regularization ............ entropy_bonus()  (조기 수렴 방지, 탐험 유지)
  5) Actor/Critic 파라미터 공유 ......... ActorCritic (공통 몸통 + 두 개의 head)
  6) RMSProp 옵티마이저 ................. main()의 optim.RMSprop (논문 실험에서 가장 안정적)
  7) Gradient norm clipping ............ main()의 clip_grad_norm_

환경: CartPole-v1 (gymnasium)
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ============================== Hyperparameters ==============================
n_envs          = 8         # 병렬 환경 수 (논문 A3C: 16개 CPU 스레드)
learning_rate   = 0.0007
gamma           = 0.99
n_steps         = 5         # n-step return의 n (논문: t_max = 5)
value_coef      = 0.5       # critic 손실 가중치
entropy_coef    = 0.01      # 엔트로피 보너스 가중치 β (논문: 0.01)
max_grad_norm   = 0.5
max_train_steps = 20000     # 업데이트 루프 횟수 (총 환경 스텝 = max_train_steps * n_envs)
print_interval  = 500
env_id          = "CartPole-v1"


# ============================================================================
# [도구 1] 병렬 actor — 동기식 벡터 환경
#   - DQN은 리플레이 버퍼로 데이터 상관관계를 깼지만, A3C/A2C는 on-policy라 버퍼를 쓸 수 없다.
#   - 대신 "서로 다른 상태에 있는 여러 환경"에서 동시에 경험을 모아 한 배치로 쓰면,
#     배치 내 샘플들이 서로 다른 궤적에서 오므로 상관관계가 자연스럽게 줄어든다.
#   - 논문의 A3C는 이를 비동기 스레드로 했고, A2C는 아래처럼 동기적으로 한 스텝씩 진행한다.
#   - minimalRL은 multiprocessing을 썼지만, 여기서는 단순화를 위해 한 프로세스에서 순차 실행.
#   - 에피소드가 끝난 환경은 즉시 reset 해서 항상 n_envs개의 환경이 돌아가게 한다.
# ============================================================================
class SyncVecEnv:
    def __init__(self, env_id, n_envs):
        self.envs = [gym.make(env_id) for _ in range(n_envs)]

    def reset(self):
        # 환경마다 seed를 달리 해 서로 다른 초기 상태에서 시작 (다양성 확보)
        return np.stack([env.reset(seed=i)[0] for i, env in enumerate(self.envs)])

    def step(self, actions):
        obs_lst, r_lst, mask_lst = [], [], []
        for env, a in zip(self.envs, actions):
            s_prime, r, terminated, truncated, _ = env.step(int(a))
            done = terminated or truncated
            if done:
                s_prime, _ = env.reset()   # 자동 리셋
            obs_lst.append(s_prime)
            r_lst.append(r)
            # mask = 0 이면 n-step return 계산 시 이후 보상/가치를 이어 붙이지 않는다.
            # (단순화: truncated도 종료로 취급. minimalRL과 동일)
            mask_lst.append(0.0 if done else 1.0)
        return (np.stack(obs_lst),
                np.array(r_lst, dtype=np.float32),
                np.array(mask_lst, dtype=np.float32))

    def close(self):
        for env in self.envs:
            env.close()


# ============================================================================
# [도구 5] Actor-Critic 네트워크 (파라미터 공유)
#   - 논문: 정책 π와 가치 V가 출력층을 제외한 모든 파라미터를 공유.
#     → 상태 표현(feature)을 같이 배우므로 표본 효율이 좋아진다.
#   - pi head: softmax로 행동 확률 출력 (Actor)
#   - v  head: 스칼라 상태 가치 출력 (Critic)
# ============================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)       # 공유 몸통
        self.fc_pi = nn.Linear(256, n_actions)   # Actor head
        self.fc_v = nn.Linear(256, 1)            # Critic head

    def pi(self, x, softmax_dim=1):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        return F.softmax(x, dim=softmax_dim)

    def v(self, x):
        x = F.relu(self.fc1(x))
        return self.fc_v(x)


# ============================================================================
# [도구 2] n-step return (부트스트랩된 다단계 반환값)
#   - 논문 Algorithm S3 (A3C pseudo code):
#         R ← V(s_{t_max})                (마지막 상태에서 부트스트랩, 종료면 0)
#         for i = t_max-1 ... 0:  R ← r_i + γ R
#   - 1-step TD 타깃(r + γV(s'))보다 실제 보상을 더 많이 반영 → 편향↓,
#     몬테카를로(에피소드 끝까지)보다는 짧음 → 분산↓. 즉 둘 사이의 절충.
#   - 또한 보상이 한 번에 n 스텝 뒤까지 전파되어 학습이 빨라진다.
#   - mask가 0인 지점(에피소드 종료)에서는 뒤쪽 R을 끊는다.
#   입력:  v_final (N,), r_lst/mask_lst: 길이 T 리스트, 각 원소 (N,)
#   출력:  (T, N) 각 시점·각 환경의 n-step return
# ============================================================================
def compute_nstep_returns(v_final, r_lst, mask_lst):
    G = v_final
    returns = []
    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        returns.append(G)
    returns.reverse()
    return np.stack(returns)


# ============================================================================
# [도구 4] Entropy regularization
#   - 논문: 목적함수에 β·H(π(·|s))를 더함.
#     "improved exploration by discouraging premature convergence to suboptimal deterministic policies"
#   - 엔트로피가 높을수록 정책이 여러 행동에 확률을 고루 나눔 → 탐험 유지.
#   - 손실을 최소화하므로 loss에서는 "- β · H" 형태로 들어간다.
# ============================================================================
def entropy_bonus(dist):
    return dist.entropy().mean()


# ============================================================================
# A2C 손실 함수 (Actor + Critic + Entropy)
#   L = L_policy + c_v · L_value - β · H
#   L_policy = - E[ log π(a|s) · A ]          (A는 상수 취급: detach)
#   L_value  =   E[ (R - V(s))² ]
#   [도구 3] Advantage A = R^(n) - V(s): baseline을 빼서 policy gradient의 분산 감소
# ============================================================================
def compute_a2c_loss(model, s_lst, a_lst, td_target, obs_dim):
    s_vec = torch.tensor(np.array(s_lst), dtype=torch.float).view(-1, obs_dim)   # (T*N, obs)
    a_vec = torch.tensor(np.array(a_lst)).view(-1)                                # (T*N,)
    td_target_vec = torch.tensor(td_target, dtype=torch.float).view(-1)          # (T*N,)

    v = model.v(s_vec).view(-1)
    advantage = td_target_vec - v

    dist = Categorical(model.pi(s_vec, softmax_dim=1))
    log_prob_a = dist.log_prob(a_vec)

    # Actor: advantage.detach() → critic 쪽 gradient가 actor 손실로 섞이지 않게
    policy_loss = -(log_prob_a * advantage.detach()).mean()
    # Critic: n-step return을 타깃으로 하는 회귀
    value_loss = advantage.pow(2).mean()
    entropy = entropy_bonus(dist)

    return policy_loss + value_coef * value_loss - entropy_coef * entropy


def test(step_idx, model, num_test=10):
    env = gym.make(env_id)
    score = 0.0
    for _ in range(num_test):
        s, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                prob = model.pi(torch.from_numpy(s).float(), softmax_dim=0)
            a = Categorical(prob).sample().item()
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            score += r
    print(f"Step # : {step_idx}, avg score : {score / num_test:.1f}")
    env.close()


def main():
    envs = SyncVecEnv(env_id, n_envs)
    obs_dim = envs.envs[0].observation_space.shape[0]
    n_actions = envs.envs[0].action_space.n

    model = ActorCritic(obs_dim, n_actions)
    # [도구 6] 논문은 SGD/RMSProp/shared RMSProp을 비교했고, RMSProp 계열이 가장 견고했다.
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-5)

    s = envs.reset()
    for step_idx in range(1, max_train_steps + 1):
        s_lst, a_lst, r_lst, mask_lst = [], [], [], []

        # --- 1) 롤아웃: n_envs개 환경에서 n_steps만큼 현재 정책으로 데이터 수집 (on-policy) ---
        for _ in range(n_steps):
            with torch.no_grad():
                prob = model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().numpy()
            s_prime, r, mask = envs.step(a)

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            mask_lst.append(mask)
            s = s_prime

        # --- 2) 마지막 상태의 가치로 부트스트랩하여 n-step return 계산 ---
        with torch.no_grad():
            v_final = model.v(torch.from_numpy(s).float()).view(-1).numpy()
        td_target = compute_nstep_returns(v_final, r_lst, mask_lst)

        # --- 3) 수집한 배치(T * N 샘플)로 한 번 업데이트 ---
        loss = compute_a2c_loss(model, s_lst, a_lst, td_target, obs_dim)
        optimizer.zero_grad()
        loss.backward()
        # [도구 7] gradient clipping: on-policy 업데이트가 한 번에 너무 크게 바뀌는 것 방지
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if step_idx % print_interval == 0:
            test(step_idx, model)

    envs.close()


if __name__ == "__main__":
    main()

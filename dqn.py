"""
DQN (Deep Q-Network) — minimalRL 스타일 단일 파일 구현

논문: Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015
     (선행 워크숍 논문: "Playing Atari with Deep Reinforcement Learning", 2013)

[알고리즘 핵심 아이디어]
  Q-learning의 Bellman 최적 방정식
        Q*(s, a) = E[ r + γ · max_a' Q*(s', a') ]
  을 신경망 Q(s, a; θ)로 근사한다. 그런데 신경망 + 부트스트래핑 + off-policy를 함께 쓰면
  학습이 쉽게 발산하기 때문에(“deadly triad”), 논문은 아래 도구들로 학습을 안정화한다.

[논문에서 성능/안정성을 위해 도입한 도구 → 이 파일의 함수]
  1) Experience Replay ............ ReplayBuffer  (데이터 상관관계 제거, 샘플 재사용)
  2) Target Network ............... hard_update_target()  (타깃이 매 스텝 흔들리는 문제 완화)
  3) Error clipping (Huber loss) .. huber_loss()  (TD 오차가 커도 gradient 크기를 1로 제한)
  4) Reward clipping .............. clip_reward()  (게임마다 다른 보상 스케일을 [-1, 1]로 통일)
  5) ε-greedy 선형 감소 ........... linear_epsilon()  (1.0 → 0.1 로 선형 감소)
  6) Gradient norm clipping ....... train() 안의 clip_grad_norm_ (Huber와 같은 목적의 실무적 보강)
  ※ Atari 전용 전처리(프레임 스킵, 4프레임 스택, 84x84 그레이스케일)는 이미지 환경용이라
    CartPole(벡터 관측)에서는 생략했다.

환경: CartPole-v1 (gymnasium API: reset → (obs, info), step → (obs, r, terminated, truncated, info))
"""
import collections
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================== Hyperparameters ==============================
learning_rate       = 0.0005
gamma               = 0.98      # 할인율 γ
buffer_limit        = 50000     # 리플레이 버퍼 크기 (논문: 1,000,000)
batch_size          = 32        # 논문과 동일
warmup_steps        = 2000      # 버퍼가 이만큼 찬 뒤에 학습 시작 (논문: replay start size 50,000)
train_freq          = 1         # 몇 환경 스텝마다 1번 업데이트할지 (논문: 4)
target_update_steps = 500       # 타깃 네트워크 동기화 주기 C (논문: 10,000 업데이트)
eps_start           = 1.0       # 초기 ε (논문: 1.0)
eps_end             = 0.01      # 최종 ε (논문: 0.1)
eps_decay_steps     = 10000     # ε을 선형 감소시키는 구간 (논문: 1,000,000 프레임)
max_grad_norm       = 10.0
use_reward_clipping = True
max_episodes        = 1000
print_interval      = 20


# ============================================================================
# [도구 1] Experience Replay
#   - 에이전트의 경험 (s, a, r, s', done_mask)을 버퍼에 저장해 두고,
#     학습할 때는 무작위로 미니배치를 뽑는다.
#   - 효과 ①: 연속된 샘플 간의 강한 시간적 상관관계를 깨서 SGD의 i.i.d. 가정에 가깝게 만든다.
#   - 효과 ②: 한 번 얻은 경험을 여러 번 재사용 → 데이터 효율 향상.
#   - 효과 ③: 현재 정책이 만든 데이터에만 끌려가며 생기는 피드백 루프(진동/발산)를 완화.
#   - Q-learning은 off-policy이므로 과거 정책이 만든 데이터로 학습해도 이론적으로 문제없다.
# ============================================================================
class ReplayBuffer:
    def __init__(self):
        # deque(maxlen=N): 가득 차면 가장 오래된 경험부터 자동으로 버린다 (FIFO)
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        # 균등 무작위 샘플링 (논문은 uniform. 이후 PER 논문에서 우선순위 샘플링으로 확장됨)
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for s, a, r, s_prime, done_mask in mini_batch:
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (torch.tensor(np.array(s_lst), dtype=torch.float),
                torch.tensor(a_lst),                                  # (B, 1) int64 → gather용
                torch.tensor(r_lst, dtype=torch.float),               # (B, 1)
                torch.tensor(np.array(s_prime_lst), dtype=torch.float),
                torch.tensor(done_mask_lst, dtype=torch.float))       # (B, 1) 종료면 0, 아니면 1

    def size(self):
        return len(self.buffer)


# ============================================================================
# Q-네트워크 Q(s, ·; θ)
#   - 논문 구조의 핵심: 상태 s를 입력으로 받아 "모든 행동의 Q값"을 한 번의 forward로 출력.
#     (s, a)를 입력으로 받는 구조보다 max_a Q(s, a)를 계산하기가 훨씬 저렴하다.
#   - 논문은 CNN(3 conv + 1 FC)을 썼지만, CartPole은 4차원 벡터이므로 MLP로 대체.
# ============================================================================
class Qnet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2):
        super().__init__()
        self.n_actions = n_actions
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)   # 행동 개수만큼 Q값 출력

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sample_action(self, obs, epsilon):
        # ε-greedy 행동 선택 (논문 Algorithm 1: "With probability ε select a random action")
        #   - 확률 ε  : 무작위 행동 → 탐험(exploration)
        #   - 확률 1-ε: argmax_a Q(s, a) → 활용(exploitation)
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            return self.forward(obs).argmax().item()


# ============================================================================
# [도구 5] ε 선형 감소 스케줄
#   - 논문: 처음 100만 프레임 동안 ε을 1.0 → 0.1로 선형 감소, 이후 0.1 고정.
#   - 초기엔 Q값이 엉망이므로 거의 무작위로 탐험하고, 학습이 진행될수록 greedy하게 행동.
# ============================================================================
def linear_epsilon(step):
    frac = min(step / eps_decay_steps, 1.0)
    return eps_start + frac * (eps_end - eps_start)


# ============================================================================
# [도구 4] Reward clipping
#   - 논문: "all positive rewards to be 1 and all negative rewards to be −1"
#   - 게임마다 점수 스케일이 달라도 같은 학습률/하이퍼파라미터로 학습할 수 있게 한다.
#   - 단점: 보상의 크기 정보를 잃는다 (큰 보상과 작은 보상을 구분 못함).
#   - CartPole은 보상이 항상 +1이라 결과에 영향은 없지만, 논문 구성 요소로 포함.
# ============================================================================
def clip_reward(r):
    return float(np.sign(r)) if use_reward_clipping else float(r)


# ============================================================================
# TD 타깃 계산 (Q-learning의 Bellman 최적 backup)
#       y = r + γ · max_a' Q(s', a'; θ⁻) · (1 - done)
#   - θ⁻ : 타깃 네트워크의 파라미터 (도구 2). 현재 네트워크 θ가 아니다!
#   - done_mask가 0이면(진짜 종료 상태) 미래 가치를 더하지 않는다.
#   - torch.no_grad(): 타깃은 "정답 레이블"처럼 취급 → 타깃 쪽으로는 gradient를 흘리지 않음
#     (semi-gradient 방식).
# ============================================================================
def compute_td_target(q_target, r, s_prime, done_mask):
    with torch.no_grad():
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)   # (B, 1)
    return r + gamma * max_q_prime * done_mask


# ============================================================================
# [도구 3] Error clipping = Huber loss
#   - 논문: "clip the error term from the update r + γ max Q − Q to be between −1 and 1"
#   - 이것은 손실 함수 관점에서 Huber loss (|δ|≤1 에서는 ½δ², 그 밖에서는 |δ|-½)와 같다.
#   - TD 오차가 아주 클 때 MSE는 gradient도 커져 학습이 튀는데,
#     Huber는 gradient 크기를 최대 1로 제한해 안정성을 높인다.
# ============================================================================
def huber_loss(q_a, target):
    return F.smooth_l1_loss(q_a, target)


# ============================================================================
# [도구 2] Target network (hard update)
#   - 논문: "Every C updates we clone the network Q to obtain a target network Q̂"
#   - 타깃 y가 매 업데이트마다 같이 움직이면 "움직이는 과녁을 쫓는" 상황이 되어 진동/발산.
#   - C 스텝 동안 타깃 네트워크를 고정 → 그 동안은 지도학습처럼 안정적인 회귀 문제가 된다.
# ============================================================================
def hard_update_target(q, q_target):
    q_target.load_state_dict(q.state_dict())


# ============================================================================
# 한 번의 학습 업데이트 (논문 Algorithm 1의 gradient descent 단계)
#   L(θ) = E_{(s,a,r,s')~D} [ Huber( y - Q(s, a; θ) ) ]
# ============================================================================
def train(q, q_target, memory, optimizer):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    q_out = q(s)                    # (B, n_actions): 모든 행동의 Q값
    q_a = q_out.gather(1, a)        # (B, 1): 실제로 취한 행동 a의 Q(s, a)만 선택

    target = compute_td_target(q_target, r, s_prime, done_mask)
    loss = huber_loss(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    # [도구 6] gradient norm clipping: Huber loss와 같은 취지로 큰 업데이트를 방지
    nn.utils.clip_grad_norm_(q.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()


def main():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q = Qnet(obs_dim, n_actions)
    q_target = Qnet(obs_dim, n_actions)
    hard_update_target(q, q_target)      # 시작 시 두 네트워크를 동일하게
    memory = ReplayBuffer()

    # 논문은 RMSProp을 사용했지만, minimalRL과 같이 Adam 사용 (실무에서 더 흔함)
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    total_steps = 0
    score = 0.0

    for n_epi in range(max_episodes):
        s, _ = env.reset()
        done = False

        while not done:
            epsilon = linear_epsilon(total_steps)
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # done_mask: "진짜 종료(terminated)"일 때만 0.
            #   시간 제한으로 끊긴 경우(truncated)는 실제로는 미래가 있으므로 부트스트랩을 유지한다.
            done_mask = 0.0 if terminated else 1.0
            memory.put((s, a, clip_reward(r), s_prime, done_mask))

            s = s_prime
            score += r
            total_steps += 1

            # 버퍼가 충분히 찬 뒤(warm-up)부터 train_freq 스텝마다 업데이트
            if memory.size() > warmup_steps and total_steps % train_freq == 0:
                train(q, q_target, memory, optimizer)

            # C 스텝마다 타깃 네트워크 동기화
            if total_steps % target_update_steps == 0:
                hard_update_target(q, q_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"n_episode: {n_epi}, avg score: {score / print_interval:.1f}, "
                  f"buffer: {memory.size()}, eps: {epsilon * 100:.1f}%")
            score = 0.0

    env.close()


if __name__ == "__main__":
    main()

# Minimal RL in PyTorch: DQN, A2C, PPO, SAC (한국어 주석)

**강화학습 알고리즘 DQN, A2C, PPO, SAC를 PyTorch로 짧게 구현한 코드**입니다.
[minimalRL](https://github.com/seungeunrho/minimalRL)처럼 알고리즘 하나당 파일 하나로 작성했고,
각 논문에서 성능을 높이려고 쓴 기법을 **함수로 분리**한 뒤 **한국어 주석으로 자세히 설명**했습니다.

Minimal, single-file PyTorch implementations of **DQN, A2C, PPO and SAC** with detailed Korean comments.
Each paper's key tricks (experience replay, target network, GAE, clipped surrogate objective,
twin Q, automatic entropy tuning, etc.) are written as separate, readable functions.

키워드: 강화학습, 심층 강화학습, 딥러닝, 파이토치, Deep Reinforcement Learning, PyTorch, gymnasium,
DQN 구현, A2C 구현, PPO 구현, SAC 구현, 강화학습 코드, 강화학습 예제, minimalRL

## 파일 구성

| 파일 | 알고리즘 | 환경 | 논문 | 함수로 분리한 핵심 기법 |
|---|---|---|---|---|
| [`dqn.py`](dqn.py) | DQN (Deep Q-Network) | CartPole-v1 | Mnih et al., Nature 2015 | Experience Replay, Target Network, Huber loss(error clipping), Reward clipping, ε-greedy 선형 감소, Gradient clipping |
| [`a2c.py`](a2c.py) | A2C (Advantage Actor-Critic) | CartPole-v1 | Mnih et al., ICML 2016 (A3C) | 병렬 환경, n-step return, Advantage(baseline), Entropy regularization, Actor/Critic 파라미터 공유, RMSProp |
| [`ppo.py`](ppo.py) | PPO (Proximal Policy Optimization) | CartPole-v1 | Schulman et al., 2017 | Clipped surrogate objective, GAE(λ), K epoch × 미니배치, Value + Entropy 결합 손실, 학습률 선형 감소, Advantage 정규화, Value clipping |
| [`sac.py`](sac.py) | SAC (Soft Actor-Critic) | Pendulum-v1 | Haarnoja et al., 2018 | Twin Q (Clipped double-Q), Polyak soft update, Reparameterization trick, Tanh squashing log-prob 보정, 자동 엔트로피 온도 α 조절, Reward scaling |

## 실행 방법

```bash
pip install torch gymnasium
python dqn.py
python a2c.py
python ppo.py
python sac.py
```

`gymnasium` API(`terminated` / `truncated` 구분)를 사용합니다.
시간 제한으로 에피소드가 끊긴 경우(truncated)는 부트스트랩을 유지하도록 처리했습니다.

## 참고 논문

- DQN: Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015
- A3C/A2C: Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*, ICML 2016
- PPO: Schulman et al., *Proximal Policy Optimization Algorithms*, 2017
- GAE: Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016
- SAC: Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018
- SAC v2: Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, 2018

## 참고 코드

- [seungeunrho/minimalRL](https://github.com/seungeunrho/minimalRL)

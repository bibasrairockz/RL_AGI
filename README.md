# Deep Q-Network (DQN) with PyTorch Lightning

## Overview
This project implements a **Deep Q-Network (DQN)** using **PyTorch Lightning** for training an agent to solve **OpenAI Gym environments**. The implementation follows best practices, including experience replay, target networks, and an epsilon-greedy policy.  
![Screenshot 2025-03-10 133309](https://github.com/user-attachments/assets/64b8789b-8a52-4217-8f2f-6364a5803f89)    
  

https://github.com/user-attachments/assets/d81fb7cc-5b55-4a11-badb-d8fc838823ee



## Features
- Uses **PyTorch Lightning** for structured model training
- Implements **experience replay** for stable learning
- Uses a **target network** to reduce instability in Q-learning
- Supports **epsilon-greedy exploration** with decay
- Includes **TensorBoard logging** for easy visualization
- Supports **video recording** of agent performance

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Training the DQN Agent
Run the following command to train the agent:
```bash
python train.py
```

### Testing the Trained Agent
Once training is complete, you can evaluate the model:
```bash
python test.py
```

## Project Structure
```
├── dqn.py              # DQN Model Implementation
├── replay_buffer.py    # Experience Replay Buffer
├── train.py            # Training Script
├── test.py             # Testing Script
├── utils.py            # Helper Functions
├── README.md           # Project Documentation
└── requirements.txt    # Dependencies
```

## Implementation Details
- **Neural Network Architecture**: A simple feedforward network with fully connected layers and ReLU activation.
- **Experience Replay**: Stores past experiences and samples mini-batches to break correlation in training data.
- **Target Network**: A separate Q-network is updated periodically to stabilize training.
- **Exploration Strategy**: Uses an epsilon-greedy policy with exponential decay.
- **Optimization**: Adam optimizer with Huber loss for stable training.

## Hyperparameters
| Hyperparameter  | Value |
|---------------|-------|
| Learning Rate | 0.001 |
| Batch Size    | 64    |
| Gamma (γ)    | 0.99  |
| Target Sync Rate | 100 episodes |
| Epsilon Decay | Exponential |

## Visualization
Training progress, rewards, and losses can be visualized using TensorBoard:
```bash
tensorboard --logdir=lightning_logs/
```

## Future Improvements
- Implement **Prioritized Experience Replay (PER)**
- Use **Double DQN** for better performance
- Add support for **continuous action spaces** using DDPG

## Acknowledgments
- **PyTorch Lightning** for structured deep learning
- **OpenAI Gym** for reinforcement learning environments

## License
This project is licensed under the MIT License.

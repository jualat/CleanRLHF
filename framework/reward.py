import torch
import torch.nn as nn

class RewardNet(nn.Module):
    def __init__(self, obs_space_dim, action_space_dim, hidden_dim):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(obs_space_dim + action_space_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def preference_prob(self, r1, r2):
        # Probability based on Bradley-Terry model
        # r_{1,2} shape: (num_steps,)
        exp1 = torch.exp(torch.sum(r1))
        exp2 = torch.exp(torch.sum(r2))
        prop1 = exp1 / (exp1 + exp2)
        prop2 = exp2 / (exp1 + exp2)
        return prop1, prop2

    def preference_loss(self, predictions, preferences):
        # Compute loss based on human feedback
        # predictions shape: (batch_size, 2)
        # preferences shape: (batch_size, 2)
        return -torch.sum(preferences * torch.log(predictions))

def train(model, optimizer, writer, global_step, D_s, D_mu, epochs, batch_size):
    # D_s shape: (num_queries, 2, num_steps, obs_space_dim + action_space_dim)
    # D_mu shape: (num_queries)

    num_queries = D_s.size(0)

    for epoch in range(epochs):
        indices = torch.randperm(num_queries)[:batch_size]

        for i in indices:
            optimizer.zero_grad()

            trajectory1 = D_s[i][0]
            trajectory2 = D_s[i][1]

            r1 = model(trajectory1)  # shape (num_steps,)
            r2 = model(trajectory2)  # shape (num_steps,)

            prob1, prob2 = model.preference_prob(r1, r2)
            predictions = torch.stack([prob1, prob2])
            preferences = D_mu[i]

            loss = model.preference_loss(predictions, preferences)
            loss.backward()
            optimizer.step()

            writer.add_scalar("losses/reward_loss", loss.item(), global_step)

        if epoch % 100 == 0:
            print(f"Reward epoch {epoch}, Loss {loss.item()}")

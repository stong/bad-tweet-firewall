import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import math
import random


# Load our data
with open('tweets_export.json', 'r') as f:
    tweets = json.load(f)

with open('tweet_embeddings.json', 'r') as f:
    embeddings = json.load(f)

with open('ratings_export.json', 'r') as f:
    ratings = json.load(f)

# Convert ratings to dict for easier lookup
ratings_dict = {r['tweetId']: r['rating'] for r in ratings}

# Filter tweets and create our dataset
filtered_data = []
for tweet in tweets:
    if tweet['tweetId'] in embeddings and len(tweet['text']) > 0:
        rating = ratings_dict.get(tweet['tweetId'], 0)  # 0 for unrated
        # Convert ratings to target values: 1 -> 1.0, 2 -> -1.0, unrated -> 0.0
        target = 0.0 if rating == 0 else (-1.0 if rating == 2 else 1.0)
        if target == 1.0 or tweet['page'] == "https://x.com/home":
            filtered_data.append({
                'embedding': embeddings[tweet['tweetId']],
                'target': target
            })

# uhhh ok so a problem is there's a lot more negative samples than positive which fucks up the training
# so we shape the training data so it's about 33% positive samples, 66% negative samples, and all of the negative samples we marked as "definitely safe" ('2')

positive = [x for x in filtered_data if x['target']==1.0]
negative = [x for x in filtered_data if x['target']==0.0]
negative_force = [x for x in filtered_data if x['target']==-1.0]
assert len(positive)+len(negative)+len(negative_force)==len(filtered_data)
for x in negative_force:
    x['target'] = 0.0

print('+ samples',len(positive))
print('- samples',len(negative))
nsamples = min(len(positive), len(negative))

random.shuffle(positive)
random.shuffle(negative)
positive = positive[:nsamples]
negative = negative[:int(1.5*nsamples)]
print('negative:',len(negative) + len(negative_force), f'({len(negative_force)} special)')
print(f'positive:{len(positive)}')
filtered_data = [] + negative_force + positive + negative
random.shuffle(filtered_data)
print('total',len(filtered_data))

# Create PyTorch Dataset
class TweetDataset(Dataset):
    def __init__(self, data):
        self.embeddings = torch.tensor([d['embedding'] for d in data], dtype=torch.float32)
        self.targets = torch.tensor([d['target'] for d in data], dtype=torch.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]

# Create model
class TweetRegressor(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # 0 to 1
        )
    
    def forward(self, x):
        return self.model(x)

# Split data
np.random.shuffle(filtered_data)
split = int(len(filtered_data) * 0.8)
train_data = filtered_data[:split]
val_data = filtered_data[split:]

# Create data loaders
train_dataset = TweetDataset(train_data)
val_dataset = TweetDataset(val_data)
BATCH_SIZE=len(train_dataset)
print('batch size:', BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# Initialize model, loss, and optimizer
model = TweetRegressor()
criterion = nn.MSELoss()

# Training loop
num_epochs = 3000
warmup_percentage = 0.25
total_steps = num_epochs * len(train_loader)
warmup_steps = int(total_steps * warmup_percentage)

# Learning rate scheduler with linear warmup and cosine decay
def lr_lambda(current_step):
    if current_step < warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
scheduler = LambdaLR(optimizer, lr_lambda)

best_val_loss = float('inf')
best_model = None
step = 0

for epoch in tqdm(range(num_epochs)):
    # Training
    model.train()
    train_loss = 0
    for embeddings, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        step += 1
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for embeddings, targets in val_loader:
            outputs = model(embeddings)
            val_loss += criterion(outputs, targets).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    if epoch % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}, step: {step}')
    
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()

    if val_loss >= best_val_loss*1.02:
        print("early stopping due to exceeded best val loss")
        break

# Save the model
torch.save(best_model, 'tweet_regressor.pt')


# eval
model.eval()
true_positives = 0
false_positives = 0
false_negatives = 0

with torch.no_grad():
    for embeddings, targets in val_loader:
        outputs = model(embeddings)
        predictions = torch.where(outputs > 0.5, 1.0, 0.0)
        
        true_positives += ((predictions == 1) & (targets == 1)).sum().item()
        false_positives += ((predictions == 1) & (targets == 0)).sum().item()
        false_negatives += ((predictions == 0) & (targets == 1)).sum().item()

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nEvaluation Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
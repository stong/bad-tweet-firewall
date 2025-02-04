import torch
import torch.nn as nn

# First recreate the model architecture
class TweetRegressor(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # -1 to 1
        )
    
    def forward(self, x):
        return self.model(x)

# Create model instance
model = TweetRegressor()

# Load the saved state dict
model.load_state_dict(torch.load('tweet_regressor.pt'))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 256)
torch.onnx.export(model, 
                 dummy_input, 
                 "Source/tweet_regressor.onnx",
                 export_params=True,
                 opset_version=12,
                 do_constant_folding=True,
                 input_names=['input'],
                 output_names=['output'],
                 dynamic_axes={'input': {0: 'batch_size'},
                             'output': {0: 'batch_size'}})

print("Model exported to Source/tweet_regressor.onnx")
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.5):
        """
        input_dim: BoW 特征维度（CountVectorizer.max_features）
        hidden_dim: 第一隐藏层大小
        dropout: Dropout 比例
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 输出两类 logits
        )

    def forward(self, x):
        return self.net(x)
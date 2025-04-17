import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from dataset import EmailDataset
from model import MLPClassifier

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds += preds
        all_labels += y.cpu().tolist()

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device)

        logits = model(x)
        probs = F.softmax(logits, dim=1)[:,1]
        preds = logits.argmax(dim=1)

        all_probs += probs.cpu().tolist()
        all_preds += preds.cpu().tolist()
        all_labels += y.cpu().tolist()

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    return acc, prec, rec, f1, auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('checkpoints', exist_ok=True)

    # 加载数据集
    phishing = EmailDataset('../data/phishing',    label=1)
    legit    = EmailDataset('../data/legitimate_sample', label=0,
                             vectorizer=phishing.vectorizer)
    full_ds = ConcatDataset([phishing, legit])

    # 划分训练/验证
    train_size = int(0.8 * len(full_ds))
    val_size   = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)

    # 构建模型与优化器
    input_dim = phishing.features.shape[1]
    model = MLPClassifier(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_f1 = 0.0
    for epoch in range(1, 11):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_prec, val_rec, val_f1, val_auc = eval_epoch(model, val_loader, device)

        print(f"[Epoch {epoch:02d}] "
              f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
              f"Val acc={val_acc:.4f}, prec={val_prec:.4f}, rec={val_rec:.4f}, f1={val_f1:.4f}, auc={val_auc:.4f}")

        # 保存最优模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'checkpoints/best_mlp.pth')
            print("  → Saved new best model")

if __name__ == "__main__":
    main()
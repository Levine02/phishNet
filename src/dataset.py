import os, mailbox, re
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer

class EmailDataset(Dataset):
    def __init__(self, root_dir, label, vectorizer=None, max_features=5000):
        self.emails, self.labels = [], []
        self.vectorizer = vectorizer or CountVectorizer(max_features=max_features)

        # 1) 先检查 .mbox 文件
        for fname in os.listdir(root_dir):
            if fname.endswith('.mbox'):
                mbox = mailbox.mbox(os.path.join(root_dir, fname))
                for msg in mbox:
                    text = self._extract_text(msg)
                    if text:
                        self.emails.append(text); self.labels.append(label)

        # 2) 再递归扫描 maildir（或者任意子文件夹下的文件）
        for dirpath, _, files in os.walk(root_dir):
            for fname in files:
                path = os.path.join(dirpath, fname)
                # 跳过已经处理过的 .mbox
                if fname.endswith('.mbox'):
                    continue
                # 假设其余文件都是单封邮件，按 email 库解析
                try:
                    from email import policy, message_from_file
                    with open(path, encoding='utf-8', errors='ignore') as f:
                        msg = message_from_file(f, policy=policy.default)
                    text = self._extract_text(msg)
                    if text:
                        self.emails.append(text); self.labels.append(label)
                except:
                    pass

        # Fit / transform
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit(self.emails)
        self.features = self.vectorizer.transform(self.emails)

    def _extract_text(self, msg):
        if msg.is_multipart():
            parts = []
            for p in msg.walk():
                if p.get_content_type()=='text/plain':
                    payload = p.get_payload(decode=True)
                    if payload: parts.append(payload)
            raw = b''.join(parts).decode('utf-8', 'ignore')
        else:
            raw = msg.get_payload(decode=True).decode('utf-8', 'ignore')
        return re.sub(r'<[^>]+>', ' ', raw)

    def __len__(self):   return len(self.labels)
    def __getitem__(self,i):
        x = self.features[i].toarray().squeeze(0)
        return x, self.labels[i]
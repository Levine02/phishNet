import os
import re
import mailbox
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer

class EmailDataset(Dataset):
    def __init__(self, root_dir, label, vectorizer=None, max_features=5000):
        """
        root_dir: 存放 mbox 或 .eml 文件的目录
        label: 1 表示钓鱼邮件，0 表示正常邮件
        vectorizer: sklearn CountVectorizer，共享词典
        max_features: 词袋维度
        """
        self.emails = []
        self.labels = []
        self.vectorizer = vectorizer or CountVectorizer(max_features=max_features)

        # 遍历所有 mbox 文件
        for fname in os.listdir(root_dir):
            if not fname.endswith('.mbox'):
                continue
            path = os.path.join(root_dir, fname)
            mbox = mailbox.mbox(path)
            for msg in mbox:
                text = self._extract_text(msg)
                if text and text.strip():
                    self.emails.append(text)
                    self.labels.append(label)

        # 构建或复用词典
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit(self.emails)
        # 文本转稀疏特征矩阵
        self.features = self.vectorizer.transform(self.emails)

    def _extract_text(self, msg):
        # 提取纯文本部分，并去除 HTML 标签
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        parts.append(payload)
            raw = b''.join(parts).decode('utf-8', errors='ignore')
        else:
            raw = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        # 去掉 HTML 标签
        return re.sub(r'<[^>]+>', ' ', raw)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回 numpy 数组和标签
        x = self.features[idx].toarray().squeeze(0)
        y = self.labels[idx]
        return x, y
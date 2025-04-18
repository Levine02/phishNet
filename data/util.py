import os, random, shutil

src_root = 'legitimate/maildir'
dst_root = 'legitimate_sample'

# 1) 收集所有邮件文件的绝对路径
all_files = []
for dirpath, _, files in os.walk(src_root):
    for fname in files:
        all_files.append(os.path.join(dirpath, fname))

# 2) 随机抽 5000 封（你可以改成 3000 或 10000）
sample_num = 5000
sample_files = random.sample(all_files, min(sample_num, len(all_files)))

# 3) 拷贝到新目录，保持原有子目录结构
for src_path in sample_files:
    rel = os.path.relpath(src_path, src_root)
    dst_path = os.path.join(dst_root, rel)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)

print(f"抽样完毕，已拷贝 {len(sample_files)} 封邮件到 {dst_root}")
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# 路径设置
data_dir = '/raid/dzz/art/train'
csv_file = '/raid/dzz/art/train.csv'
output_dir = '/raid/dzz/art/art_dataset'


train_output_dir = os.path.join(output_dir, 'train')
val_output_dir = os.path.join(output_dir, 'val')
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

df = pd.read_csv(csv_file)


train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])


def copy_files(df, output_dir):
    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, str(row['filename']) + '.jpg')
        class_dir = os.path.join(output_dir, str(row['label']))
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy(img_path, class_dir)


copy_files(train_df, train_output_dir)


copy_files(val_df, val_output_dir)

print("Finsh.")
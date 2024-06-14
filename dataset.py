import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Note: You should write a DataLoader suitable for your own Dataset!!!
class SimpleDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.root = root
        self.mode = mode
        self.image_dir = os.path.join(self.root, 'data')
        folders = sorted(os.listdir(self.image_dir))
        self.image_list = [os.path.join(self.image_dir, file) for file in folders]
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.image_list[index]
        text = image.split('/')[-1]
        prompt = text.replace('_', ' ')[:-4]
        image =  Image.open(image).convert('RGB')
        # image = image.resize((512, 512))
        image = image.resize((1024, 1024))
        image = transforms.ToTensor()(image)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        if self.mode == 'train':
            p = random.uniform(0, 1)
            if p < 0.1:
                prompt = ''
        
        # normalize
        image = image * 2. - 1.

        return {"image": image, "prompt": prompt}

if __name__ == '__main__':
    train_dataset = SimpleDataset(root="./")
    print(train_dataset.__len__())

    train_data = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False)
    # B C H W
    for i, data in enumerate(train_data):
        print(i)
        print(data['image'].shape)
        print(data['prompt'])
import os, sys
sys.path.append('C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing')
from lib import *

class Covid_XRay(Dataset):
    def __init__(self, data_dir:str = 'data', split: str = 'train') -> None:
        super().__init__()
        self.data_dir = os.path.join(data_dir, split)
        self.classes = os.listdir(self.data_dir)

        self.class_id = {key: number for number, key in enumerate(self.classes)}


        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        self.list_files = []
                
        for c in self.classes:
            class_dir = os.path.join(self.data_dir, c)
            filenames = os.listdir(class_dir)
            self.list_files.extend([(os.path.join(class_dir, filename), self.class_id[c]) for filename in filenames])



    def __len__(self) -> int:
        return len(self.list_files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        
        filename = self.list_files[index]

        img = self.transform(Image.open(filename[0]).convert('RGB'))

        return img, filename[1]
    

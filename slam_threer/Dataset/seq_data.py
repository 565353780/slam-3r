import torch

from slam3r.utils.image import load_images


class SeqData():
    def __init__(
        self,
        img_dir,     # the directory of the img sequence
        img_size=224,  # only img_size=224 is supported now 
        silent=False,  
        sample_freq=1, # the frequency of the imgs to be sampled
        num_views=-1, # only take the first num_views imgs in the img_dir
        start_freq=1,  
        postfix=None,   # the postfix of the img in the img_dir(.jpg, .png, ...)
        to_tensor=False,
        start_idx=0,
    ):
        # Note that only img_size=224 is supported now.
        # Imgs will be cropped and resized to 224x224, thus losing the information in the border.
        assert img_size==224, "Sorry, only img_size=224 is supported now."

        # load imgs with sequential number.
        # Imgs in the img_dir should have number in their names to indicate the order,
        # such as frame-0031.color.png, output_414.jpg, ...
        self.imgs = load_images(img_dir, size=img_size, 
                                verbose=not silent, img_freq=sample_freq,
                                postfix=postfix, start_idx=start_idx, img_num=num_views)

        self.num_views = num_views if num_views > 0 else len(self.imgs)
        self.stride = start_freq
        self.img_num = len(self.imgs)
        if to_tensor:
            for img in self.imgs:
                img['true_shape'] = torch.tensor(img['true_shape'])
        self.make_groups()
        self.length = len(self.groups)

        if isinstance(img_dir, str):
            if img_dir[-1] == '/':
                img_dir = img_dir[:-1]
            self.scene_names = ['_'.join(img_dir.split('/')[-2:])]

    def make_groups(self):
        self.groups = []
        for start in range(0,self.img_num, self.stride):
            end = start + self.num_views 
            if end > self.img_num:
                break
            self.groups.append(self.imgs[start:end])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]

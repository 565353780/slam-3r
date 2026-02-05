import os
import torch
from typing import Optional, List

from slam_threer.Model.image2points import Image2PointsModel
from slam_threer.Model.local2world import Local2WorldModel


class Detector(object):
    def __init__(
        self,
        i2p_model_folder_path: Optional[str]=None,
        l2w_model_folder_path: Optional[str]=None,
        device: str='cuda:0',
    ) -> None:
        self.i2p_model : Image2PointsModel
        self.l2w_model : Local2WorldModel
        self.device = device

        if i2p_model_folder_path is not None and l2w_model_folder_path is not None:
            self.loadModel(i2p_model_folder_path, l2w_model_folder_path, device)
        return

    def loadModel(
        self,
        i2p_model_folder_path: str,
        l2w_model_folder_path: str,
        device: str='cuda:0',
    ) -> bool:
        self.device = device

        self.i2p_model = Image2PointsModel.from_pretrained(i2p_model_folder_path)
        self.l2w_model = Local2WorldModel.from_pretrained(l2w_model_folder_path)

        self.i2p_model.to(self.device)
        self.l2w_model.to(self.device)
        self.i2p_model.eval()
        self.l2w_model.eval()
        return True

    def detectImageFiles(
        self,
        image_file_path_list: List[str],
    ) -> Optional[dict]:
        valid_image_file_path_list = []

        for image_file_path in image_file_path_list:
            if not os.path.exists(image_file_path):
                print('[WARN][Detector::detectImageFiles]')
                print('\t image file not exist!')
                print('\t image_file_path:', image_file_path)
                continue

            valid_image_file_path_list.append(image_file_path)

        if len(valid_image_file_path_list) == 0:
            print('[ERROR][Detector::detectImageFiles]')
            print('\t all image files not exist!')
            return None

        return {}

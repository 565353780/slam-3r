import os

from slam_threer.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    i2p_model_folder_path = home + '/chLi/Model/SLAM3R/slam3r_i2p'
    l2w_model_folder_path = home + '/chLi/Model/SLAM3R/slam3r_l2w'
    data_folder_path = home + '/chLi/Dataset/GS/haizei_1_v4/'
    image_folder_path = data_folder_path + 'colmap/gs/images/'
    device = 'cuda:0'

    detector = Detector(
        i2p_model_folder_path,
        l2w_model_folder_path,
        device,
    )

    result = detector.detectImageFolder(image_folder_path)
    assert result is not None

    detector.saveResult(result, data_folder_path + 'slam3r/')
    return True

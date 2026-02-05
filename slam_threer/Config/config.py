import torch
import argparse
import numpy as np


def getConfig():
    parser = argparse.ArgumentParser(description="Inference on a scene")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument('--save_all_views', action='store_true', help='whether to save all views respectively')

# args for the whole scene reconstruction
    parser.add_argument("--keyframe_stride", type=int, default=3, 
                        help="the stride of sampling keyframes, -1 for auto adaptation")
    parser.add_argument("--initial_winsize", type=int, default=5, 
                        help="the number of initial frames to be used for scene initialization")
    parser.add_argument("--win_r", type=int, default=3, 
                        help="the radius of the input window for I2P model")
    parser.add_argument("--conf_thres_i2p", type=float, default=1.5, 
                        help="confidence threshold for the i2p model")
    parser.add_argument("--num_scene_frame", type=int, default=10, 
                        help="the number of scene frames to be selected from \
                            buffering set when registering new keyframes")
    parser.add_argument("--max_num_register", type=int, default=10, 
                        help="maximal number of frames to be registered in one go")
    parser.add_argument("--conf_thres_l2w", type=float, default=12, 
                        help="confidence threshold for the l2w model(when saving final results)")
    parser.add_argument("--num_points_save", type=int, default=2000000, 
                        help="number of points to be saved in the final reconstruction")
    parser.add_argument("--norm_input", action="store_true", 
                        help="whether to normalize the input pointmaps for l2w model")
    parser.add_argument("--save_frequency", type=int,default=3,
                        help="per xxx frame to save")
    parser.add_argument("--save_each_frame",action='store_true',default=True,
                        help="whether to save each frame to .ply")
    parser.add_argument("--video_path",type = str)
    parser.add_argument("--retrieve_freq",type = int,default=1, 
                        help="(online mode only) frequency of retrieving reference frames")
    parser.add_argument("--update_buffer_intv", type=int, default=1, 
                        help="the interval of updating the buffering set")
    parser.add_argument('--buffer_size', type=int, default=100, 
                        help='maximal size of the buffering set, -1 if infinite')
    parser.add_argument("--buffer_strategy", type=str, choices=['reservoir', 'fifo'], default='reservoir', 
                        help='strategy for maintaining the buffering set: reservoir-sampling or first-in-first-out')
    parser.add_argument("--save_online", action='store_true', 
                        help="whether to save the construct result online.")

#params for auto adaptation of keyframe frequency
    parser.add_argument("--keyframe_adapt_min", type=int, default=1, 
                        help="minimal stride of sampling keyframes when auto adaptation")
    parser.add_argument("--keyframe_adapt_max", type=int, default=20, 
                        help="maximal stride of sampling keyframes when auto adaptation")
    parser.add_argument("--keyframe_adapt_stride", type=int, default=1, 
                        help="stride for trying different keyframe stride")
    parser.add_argument("--perframe", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42, help="seed for python random")
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id, -1 for auto select')
    parser.add_argument('--save_preds', action='store_true', help='whether to save all per-frame preds')    
    parser.add_argument('--save_for_eval', action='store_true', help='whether to save partial per-frame preds for evaluation')   
    parser.add_argument("--online", action="store_true", help="whether to implement online reconstruction")

    args = parser.parse_args()
    print("using gpu: ", args.gpu_id)
    torch.cuda.set_device(f"cuda:{args.gpu_id}")
    # print(args)
    np.random.seed(args.seed)
    return args

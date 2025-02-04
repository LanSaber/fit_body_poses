python main.py --config cfg_files/fit_smplx.yaml     --data_folder data_folder    --output_folder output_folder     --visualize="True/False"    --model_folder /home/hhm/smpl_model   --vposer_ckpt V02_05

cfg_files/fit_smplx.yaml中设置参数，主要是keypoints_folder，表示openpose（BODY25）关键点所在的路径，video_directory表示对应的mp4视频文件的路径

% clc
% clear
% 
% load('/vision/vision_users/bxiong/dataset/Webscope_I4/ydata-tvsum50-v1_0/ydata-tvsum50-matlab/matlab/ydata-tvsum50.mat');
% 
% base_dir='/vision/vision_users/bxiong/dataset/tvsum_frames/video/';
% 
% 
% img_dir='/vision/vision_users/bxiong/projects/ego_context/features/motion/video_motion';
% 
% 
% 
% 
% 
% parfor i=1:50
%     
%     
%     image_dir=tvsum50(i).video;
%     img_raw_dir=fullfile('/vision/vision_users/bxiong/dataset/tvsum_frames/video/',image_dir);
%     temp_dir=dir(img_raw_dir);
%     img_num=numel(temp_dir)-2;
%     img_dir='/vision/vision_users/bxiong/projects/ego_context/features/motion/video_motion';
%     
%     img_dir=fullfile(img_dir,image_dir)
%     
%     if ~exist(img_dir,'dir')
%         mkdir(img_dir);
%     end
%     
%     mat_dir=fullfile(img_dir,'motion.mat');
%     compute_multi_flow(image_dir,1,img_num-4,4,4,mat_dir);
%    
% 
%     
% end
% 
% 
% 
% 
% 






% 
% clc
% clear
% 
% load('/vision/vision_users/bxiong/dataset/Webscope_I4/ydata-tvsum50-v1_0/ydata-tvsum50-matlab/matlab/ydata-tvsum50.mat');
% 
% base_dir='/vision/vision_users/bxiong/dataset/tvsum_frames/video/';
% 
% 
% img_dir='/vision/vision_users/bxiong/projects/ego_context/features/motion/video_motion';


% 
% motion_strength=cell(50,1);
% 
% for i=1:50
%     
%     i
%     image_dir=tvsum50(i).video;
%     img_raw_dir=fullfile('/vision/vision_users/bxiong/dataset/tvsum_frames/video/',image_dir);
%     temp_dir=dir(img_raw_dir);
%     img_num=numel(temp_dir)-2;
%     img_dir='/vision/vision_users/bxiong/projects/ego_context/features/motion/video_motion';
%     
%     img_dir=fullfile(img_dir,image_dir);
%     
%     if ~exist(img_dir,'dir')
%         mkdir(img_dir);
%     end
%     
%     mat_dir=fullfile(img_dir,'motion.mat');
%     load(mat_dir);
%     motion_mag=zeros(numel(flow),2);
%     for frame_index=1:numel(flow)
%         dx=flow{frame_index}{1};
%         dy=flow{frame_index}{2};
%         dx_mag=mean(abs(dx(:)));
%         dy_mag=mean(abs(dy(:)));
%         motion_mag(frame_index,1)=dx_mag;
%         motion_mag(frame_index,2)=dy_mag;
%     end
%     
%     motion_strength{i}=motion_mag;
%     
% end
% 
% 
% 































% 
% tic
% for i=1100:4:2000
%     show_optical_flow(i)
%     disp(i)
%     toc
%     %pause(0.01)
% end
% 


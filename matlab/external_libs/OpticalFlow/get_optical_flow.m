function [vx,vy,warpI2]=get_optical_flow(im1,im2)


alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;

para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

% this is the core part of calling the mexed dll file for computing optical flow
% it also returns the time that is needed for two-frame estimation

[vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);

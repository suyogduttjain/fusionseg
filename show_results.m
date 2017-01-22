%1 = appearance model
%2 = motion model
model_type = 1; 

if model_type == 1
	data_dir = './images/';
	input_file = './appearance_image_list.txt';
	output_file = './appearance_output_list.txt';
elseif model_type == 2
	data_dir = './motion_images/';
	input_file = './motion_image_list.txt';
	output_file = './motion_output_list.txt';
else
	return;
end

image_files    = textread(input_file,'%s');
image_prefixes = textread(output_file,'%s');
num_images = length(image_files);

for i = 1:num_images
	feature_name = [image_prefixes{i} '_blob_0.mat'];
	data = load(fullfile(data_dir, feature_name));
	raw_result = data.data;
	
	img = imread(fullfile(data_dir,image_files{i}));
	img_row = min(size(img, 1),size(raw_result,1));
	img_col = min(size(img, 2),size(raw_result,2));
	raw_result = permute(raw_result, [2 1 3]);
	
	probs = raw_result(1:img_row, 1:img_col, :);
	[~, mask] = max(probs,[],3);
	mask = logical(mask-1);
	
	figure(1);
	subplot(2,2,1);
	imshow(img);
	
	subplot(2,2,2);
	imshow(mask);
	title('Object Mask');
	
	subplot(2,2,3);
	imagesc(probs(:,:,2));
	axis image;
	title('Object Probability');
	
	pause;

end

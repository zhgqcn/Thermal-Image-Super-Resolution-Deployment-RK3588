>> % 指定图像文件夹路径
imageFolder = 'F:\TISR-main\rknn\LR_x4';

% 获取所有图像文件名
imageFiles = dir(fullfile(imageFolder, '*.jpg')); % 假设图像格式为.jpg
numImages = numel(imageFiles);

% 创建VideoWriter对象
videoFile = 'F:\TISR-main\rknn\output_video.mp4'; % 指定输出视频文件名
fps = 24; % 每秒帧数
writerObj = VideoWriter(videoFile, 'MPEG-4');
open(writerObj);

% 循环遍历所有图像文件
for i = 1:numImages
    % 读取图像
    imagePath = fullfile(imageFolder, imageFiles(i).name);
    img = imread(imagePath);
    
    % 写入帧
    writeVideo(writerObj, img);
end

% 关闭VideoWriter对象
close(writerObj);
>> 
% vidObj = VideoReader('video/CISVideo.m4v');
% get(vidObj)
% vidObj.CurrentTime = 0;
% vidWidth = vidObj.Width;
% vidHeight = vidObj.Height;
% mov = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),'colormap',[]);
% currAxes = axes;
% 
% k = 1;
% while vidObj.CurrentTime <= 5
%     mov(k).cdata = readFrame(vidObj);
% %     Frame = readFrame(vidObj);
% %     image(mov(k).cdata, 'Parent', currAxes);
% %     [C,handle] = imcontour(rgb2gray(mov(k).cdata)); % did not return contour cnt
% % %     currAxes.Visible = 'off';
% %      pause(1/(vidObj.FrameRate * 2));
%     k = k+1
% end
figure;
close all;
figure1=figure('Position', [100, 100, 1200, 1800]);
prev_GrayIamge = rgb2gray(mov(1).cdata);
k = 2;
while k < length(mov)
%     mov(k).cdata = readFrame(vidObj);
%     Frame = readFrame(vidObj);
%     image(mov(k).cdata, 'Parent', currAxes);
    subplot(2,2,1);
    imshow(mov(k).cdata);
    str = sprintf('Original Frame (%d)', k);
    title(str);
    
    subplot(2,2,2);
    GrayIamge=rgb2gray(mov(k).cdata);
    BW_Image=im2bw(GrayIamge,graythresh(GrayIamge));
    imshow(BW_Image)
    str = sprintf('BW Frame (%d)', k);
    title(str); 
    
    % removin background is not applicable for our application as there
    % likley not to be a coherent background
%     % Get rid of huge background that touches the border
%     BW_Image = imclearborder(BW_Image, 4);
%     % Display the final image.
%     subplot(2, 2, 3);
%     imshow(BW_Image);

    B = bwboundaries(~BW_Image);
    ObjectCnt(k) = length(B);
    subplot(2,2,3);
    imshow(~BW_Image)
    str = sprintf('Objects Cnt (%d)', ObjectCnt(k));
    title(str); 
    
     subplot(2,2,4);
%     [C,handle] = imcontour(rgb2gray(mov(k).cdata)); % did not return contour cnt
%     str = sprintf('Contours (%d)', length(C));
%     title(str);
%     imcontourCnt(k) = length(C);
    
    [EdgeImg, th]=edge(Diff, 'Sobel', 0.1); % 'Sobel' Prewitt
    imshow(EdgeImg);
    str = sprintf('Edge Frame (%d)', k);
    title(str); 
    
    [~, J] = bwlabel(BW_Image);
    numberOfClosedRegions(k) = J;
     
    prev_GrayIamge = GrayIamge;
    pause(0.001);
    k = k+1;
end

figure;
plot(numberOfClosedRegions);
title('numberOfClosedRegions Distribution');
xlabel('Frame');
ylabel('numberOfClosedRegions');


figure;
plot(ObjectCnt);
title('Objects Count Distribution');
xlabel('Frame');
ylabel('Objects Count');

% figure;
% plot(imcontourCnt);
% title('imcontourCnt Distribution');
% xlabel('Frame');
% ylabel('imcontourCnt');
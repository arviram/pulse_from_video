clear;
close all;
load('bandpass_iir.mat');

fileID=4;
fileName=sprintf('%d.mp4',fileID);
featuresPath=sprintf('%d_frames',fileID);
v=VideoReader(fileName);
addpath(featuresPath);

%frames=cells({}
nFrames=floor(v.FrameRate*v.Duration);
Fs=ceil(v.FrameRate);
region1=[2:7 37 41 42 40 49];
region2=[11:16 46:48 43 55];
region3=[7:9 58:60];
region4=[9:11 56:58];

%Forhead
region5=[19:22 67 68];
region6=[23:26 69 70];
region7=[22 68 23 69];
if(fileID==1)
forehead_height=87; 
elseif (fileID==2) 
forehead_height=110; 
elseif (fileID==3) 
forehead_height=150; 
elseif (fileID==4) 
forehead_height=100; 
elseif (fileID==5) 
forehead_height=120; 
else
forehead_height=150;     
end

if(fileID==1)
x_min=50;
y_min=45;
x_max=450;
y_max=525;
elseif (fileID==2) 
x_min=70;
y_min=160;
x_max=530;
y_max=700;
elseif (fileID==3) 
x_min=250;
y_min=560;
x_max=870;
y_max=1380;
elseif (fileID==4) 
x_min=540;
x_max=920;
y_min=90;
y_max=630;
elseif (fileID==5) 
x_min=430;
x_max=850;
y_min=20;
y_max=600;
else
%x_min=min(landmarks(:,1));
%y_min=min(landmarks(:,2));
%x_max=max(landmarks(:,1));
%y_max=max(landmarks(:,2)); 
x_min=540;
x_max=920;
y_min=90;
y_max=630;
end

face_extemities=[x_min y_min;
                 x_max y_min;
                 x_min y_max;
                 x_max y_max];
                          
                          
% Initial frame                         
for i=1:1
    current_frame= readFrame(v);
    size(current_frame);
    filename=sprintf('frame%d.pts',i);
    landmarks=get_facial_coordinates(filename);
    landmarks=[landmarks;
               landmarks(19,1) landmarks(19,2)-forehead_height;
               landmarks(22,1) landmarks(22,2)-forehead_height;
               landmarks(23,1) landmarks(23,2)-forehead_height;
               landmarks(26,1) landmarks(26,2)-forehead_height];           
           
    imshow(current_frame(:,:,2));
    hold on;
    image=current_frame(:,:,1);
    %plot(landmarks(:,1),landmarks(:,2),'r*', 'MarkerSize', 1);    
    region=region1;
    plot(landmarks(region,1),landmarks(region,2),'r*', 'MarkerSize', 1);    
    region_pts=[landmarks(region,1) landmarks(region,2)];
    k=convhull(landmarks(region,1),landmarks(region,2));
    region1_hull=[region_pts(k,1) region_pts(k,2)];
    plot(region_pts(k,1),region_pts(k,2),'r-');    
    region=region2;
    plot(landmarks(region,1),landmarks(region,2),'r*', 'MarkerSize', 1);    
    region_pts=[landmarks(region,1) landmarks(region,2)];
    k=convhull(landmarks(region,1),landmarks(region,2));
    plot(region_pts(k,1),region_pts(k,2),'b-');            
    region2_hull=[region_pts(k,1) region_pts(k,2)];
    region=region3;
    plot(landmarks(region,1),landmarks(region,2),'r*', 'MarkerSize', 1);    
    region_pts=[landmarks(region,1) landmarks(region,2)];
    k=convhull(landmarks(region,1),landmarks(region,2));
    plot(region_pts(k,1),region_pts(k,2),'g-');    
    region3_hull=[region_pts(k,1) region_pts(k,2)];     
    region=region4;
    plot(landmarks(region,1),landmarks(region,2),'r*', 'MarkerSize', 1);    
    region_pts=[landmarks(region,1) landmarks(region,2)];
    k=convhull(landmarks(region,1),landmarks(region,2));
    plot(region_pts(k,1),region_pts(k,2),'g-');    
    region4_hull=[region_pts(k,1) region_pts(k,2)];     
    region=region5;
    plot(landmarks(region,1),landmarks(region,2),'r*', 'MarkerSize', 1);    
    region_pts=[landmarks(region,1) landmarks(region,2)];
    k=convhull(landmarks(region,1),landmarks(region,2));
    plot(region_pts(k,1),region_pts(k,2),'g-');    
    region5_hull=[region_pts(k,1) region_pts(k,2)];     
    region=region6;
    plot(landmarks(region,1),landmarks(region,2),'r*', 'MarkerSize', 1);    
    region_pts=[landmarks(region,1) landmarks(region,2)];
    k=convhull(landmarks(region,1),landmarks(region,2));
    plot(region_pts(k,1),region_pts(k,2),'g-');    
    region6_hull=[region_pts(k,1) region_pts(k,2)];     
    region=region7;
    plot(landmarks(region,1),landmarks(region,2),'r*', 'MarkerSize', 1);    
    region_pts=[landmarks(region,1) landmarks(region,2)];
    k=convhull(landmarks(region,1),landmarks(region,2));
    plot(region_pts(k,1),region_pts(k,2),'r-');
    region7_hull=[region_pts(k,1) region_pts(k,2)];         
    k=convhull(face_extemities(:,1),face_extemities(:,2));
    plot(face_extemities(k,1),face_extemities(k,2),'g-');    
       
    %k=boundary(landmarks(region,1),landmarks(region,2));
    %plot(region_pts(k,1),region_pts(k,2),'r-');  
    roi=image(y_min:y_max-1,x_min:x_max-1);
    box_height=y_max-y_min;
    box_width=x_max-x_min;
    J = integralImage(roi);
    
    x_start=x_min;
    y_start=y_min;
    % Divide into 20x20 boxes
    n_y=box_height/20;
    n_x=box_width/20;
    n_boxes=n_x*n_y;
    x_indices=1:n_x;
    y_indices=1:n_y;
    [gridx,gridy]=meshgrid(x_indices,y_indices);
    b_box_start_points_x=(gridx-1)*20+1;
    b_box_start_points_y=(gridy-1)*20+1;
    b_box_end_points_x=gridx*20+1;
    b_box_end_points_y=gridy*20+1;
       
    % Vectorize!
    
    b_box_start_points_x=reshape(b_box_start_points_x',n_boxes,1);
    b_box_start_points_y=reshape(b_box_start_points_y',n_boxes,1);
    b_box_end_points_x=reshape(b_box_end_points_x',n_boxes,1);
    b_box_end_points_y=reshape(b_box_end_points_y',n_boxes,1);

%    b_box_top_right_x=b_box_end_points_x;
%    b_box_top_right_y=b_box_start_points_y;
%    b_box_bottom_left_x=b_box_start_points_x;
%    b_box_bottom_left_y=b_box_end_points_y;

    linear_ind_top_left=sub2ind(20*size(gridx)+1, b_box_start_points_y, b_box_start_points_x);
    linear_ind_top_right=sub2ind(20*size(gridx)+1, b_box_start_points_y, b_box_end_points_x);
    linear_ind_bottom_left=sub2ind(20*size(gridx)+1, b_box_end_points_y, b_box_start_points_x);
    linear_ind_bottom_right=sub2ind(20*size(gridx)+1, b_box_end_points_y, b_box_end_points_x);

    %linear_ind_top_left=sub2ind(size(roi), b_box_start_points_y, b_box_start_points_x);
    %linear_ind_top_right=sub2ind(size(roi), b_box_start_points_y, b_box_end_points_x);
    %linear_ind_bottom_left=sub2ind(size(roi), b_box_end_points_y, b_box_start_points_x);
    %linear_ind_bottom_right=sub2ind(size(roi), b_box_end_points_y, b_box_end_points_x);

    b_box_start_points_x=b_box_start_points_x+x_start;
    b_box_start_points_y=b_box_start_points_y+y_start;

    % Find which region each box belongs to!
    region=zeros(n_boxes,1);
    [region1,~] = inpolygon(b_box_start_points_x,b_box_start_points_y,region1_hull(:,1),region1_hull(:,2));
    region(find(region1==1))=1;   
    [region2,~] = inpolygon(b_box_start_points_x,b_box_start_points_y,region2_hull(:,1),region2_hull(:,2));
    region(find(region2==1))=2;   
    [region3,~] = inpolygon(b_box_start_points_x,b_box_start_points_y,region3_hull(:,1),region3_hull(:,2));
    region(find(region3==1))=3;   
    [region4,~] = inpolygon(b_box_start_points_x,b_box_start_points_y,region4_hull(:,1),region4_hull(:,2));
    region(find(region4==1))=4;   
    [region5,~] = inpolygon(b_box_start_points_x,b_box_start_points_y,region5_hull(:,1),region5_hull(:,2));
    region(find(region5==1))=5;   
    [region6,~] = inpolygon(b_box_start_points_x,b_box_start_points_y,region6_hull(:,1),region6_hull(:,2));
    region(find(region6==1))=6;   
    [region7,~] = inpolygon(b_box_start_points_x,b_box_start_points_y,region7_hull(:,1),region7_hull(:,2));
    region(find(region7==1))=7;   
%    wave_forms=J(b_box_end_points_y,b_box_end_points_x)+ ...
%               J(b_box_start_points_y,b_box_start_points_x) - ...
%               J(b_box_top_right_y,b_box_top_right_x) - ...
%               J(b_box_bottom_left_y,b_box_bottom_left_x);

    valid_boxes=find(region~=0);
    linear_ind_bottom_right=linear_ind_bottom_right(valid_boxes);
    linear_ind_top_left=linear_ind_top_left(valid_boxes);
    linear_ind_bottom_left=linear_ind_bottom_left(valid_boxes);
    linear_ind_top_right=linear_ind_top_right(valid_boxes);
    n_valid_boxes=size(valid_boxes,1);

    box_intensities=J(linear_ind_bottom_right)+J(linear_ind_top_left) -J(linear_ind_bottom_left)-J(linear_ind_top_right);
    
    box_intensities=box_intensities/400;
    
    pulse_waveforms=zeros(n_valid_boxes,nFrames);    
    pulse_waveforms(:,1)=box_intensities;
    
    single_pixel=zeros(1,nFrames); 
    single_pixel_x=floor(region7_hull(1,1));
    single_pixel_y=floor(region7_hull(1,2));
    single_pixel(1)=image(single_pixel_y,single_pixel_x);

    indices=find(region==7);
    plot(b_box_start_points_x(indices,1),b_box_start_points_y(indices,1),'-or');
    region_subset_x=b_box_start_points_x(indices);
    region_subset_y=b_box_start_points_y(indices);
    k=convhull(b_box_start_points_x(indices),b_box_start_points_y(indices));    
    fill(region_subset_x(k),region_subset_y(k),'blue'); 
    
    region=region(valid_boxes);
end

start_sample=31;
end_sample=90;
n_processing_samples=end_sample-start_sample+1;

for i=2:end_sample
    fprintf('\n Processing frame %d',i);
    current_frame= readFrame(v);
    image=current_frame(:,:,1);   
    single_pixel(i)=image(single_pixel_y,single_pixel_x);    
    roi=image(y_min:y_max-1,x_min:x_max-1);
    J = integralImage(roi);
    box_intensities=J(linear_ind_bottom_right)+J(linear_ind_top_left) -J(linear_ind_bottom_left)-J(linear_ind_top_right);
    box_intensities=box_intensities/400;
    pulse_waveforms(:,i)=box_intensities;
end

pulse_waveforms=pulse_waveforms(:,1:end_sample);
pulse_waveforms=bsxfun(@minus,pulse_waveforms,mean(pulse_waveforms,2));
%pulse_subset=pulse_waveforms(find(region==7),:);
pulse_subset=pulse_waveforms;

norm_factor=max(abs(pulse_subset),[],2);
non_zero_idx=find(norm_factor>=0.5);
norm_factor=norm_factor(non_zero_idx);
pulse_subset=pulse_subset(non_zero_idx,:);
pulse_subset=bsxfun(@rdivide,pulse_subset,norm_factor);

N_points=4096;

y=zeros(size(pulse_subset,1),n_processing_samples);
pulse_fft=zeros(size(pulse_subset,1),N_points);

for i=1:size(y,1)
    filter_out=filter(Hbp5,pulse_subset(i,:));
    y(i,:)=filter_out(start_sample:end_sample);
    pulse_fft(i,:)=abs(fft(y(i,:),N_points));
end

% Maximal Ratio combining
mean_waveform=mean(y);

% Find coarse estimate
mean_waveform_fft=fft(mean_waveform,N_points);
[~,coarse_index]=max(abs(mean_waveform_fft));
coarse_freq=(coarse_index-1)*(Fs)/N_points;
coarse_bpm=coarse_freq*60;

%Estimate goodness metrics
bandwidth=11; % in terms of indices
goodness=zeros(size(y,1),1);
for i=1:size(y,1)
    half_spectrum=pulse_fft(i,1:N_points/2);
    signal_strength=sum(half_spectrum(coarse_index-(bandwidth-1)/2:coarse_index+(bandwidth-1)/2).^2);
    noise_strength=sum(half_spectrum.^2)-signal_strength;
    goodness(i)=signal_strength/noise_strength;
end

% Find fine estimate
goodness=goodness/sum(goodness);
pulse_goodness=bsxfun(@times,y,goodness);

final_pulse=sum(pulse_goodness);

figure();
plot((1:n_processing_samples)/Fs,mean_waveform,'r',(1:n_processing_samples)/Fs,final_pulse,'b');

% Find fine estimate
final_waveform_fft=fft(final_pulse,N_points);
[~,fine_index]=max(abs(final_waveform_fft));
fine_freq=(fine_index-1)*(Fs)/N_points;
fine_bpm=fine_freq*60;
fprintf('\n Pulse rate is %f bpm coarse %f bpm fine',coarse_bpm,fine_bpm);

% Frequency plots
freq=(0:N_points/6-1)*(Fs)*60/N_points;
figure();
plot(freq,abs(final_waveform_fft(1:N_points/6)));

single_pixel=single_pixel(1:end_sample);
%figure();
%plot((1:end_sample)/Fs,single_pixel);
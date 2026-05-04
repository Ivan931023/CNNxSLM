% export_static_tensors.m
clc; clear all; close all;

% ---------- Beam parameter ---------- %
lambda = 447e-6; f = 300;
Beam_size = 3.45; Output_x = 0.4; Output_y = 0.1;

% ---------- Phase pattern parameter ---------- %
CCD_pixel = 2.2e-3; range = 150;
dx = 8e-3; dy = 8e-3;

pixel_Zernike = 750;
pixel_pattern = 1080;

N = round(lambda*f / (0.008*CCD_pixel),-1); % 7620
pixel = N;
Nx = N; Ny = N;

% --------- Set the Parameter of input beam ---------
x = -Nx/2*dx : dx : (Nx/2-1)*dx;
y = -Ny/2*dy : dy : (Ny/2-1)*dy;
[X, Y] = meshgrid(x, -y);

x0 = 0; y0 = 0;
sig_x = Beam_size / 4; sig_y = Beam_size / 4;
Input_beam = exp(-((X-x0).^2/2/(sig_x)^2+(Y-y0).^2/2/(sig_y)^2)) ;
Input_beam = Input_beam ./ max(Input_beam(:));
unit_power = 12000;
Input_beam_origin_power = sum(Input_beam,'all');
ratio = (unit_power * 10000) / Input_beam_origin_power;
Input_beam = Input_beam * ratio;

% -------- Grating parameter --------%
theta_deg = -90; theta_blazed = deg2rad(theta_deg);
max_phase = 255; min_phase = 0; repeat = 1; level = 12;

grat = Blazed_grating_rotate(pixel_pattern, max_phase, min_phase, level, repeat, theta_blazed);
grat = padding(grat,pixel);
Blazed_phi = 2*pi/255 * grat; % [0 2pi]

% -------- Analytic solution --------%
x1 = 8e-3 * (-pixel_pattern/2:pixel_pattern/2-1);
y1 = 8e-3 * (-pixel_pattern/2:pixel_pattern/2-1);
[X1, Y1] = meshgrid(x1,y1);

a = Beam_size / (2*sqrt(2));
ax1 = a; ax2 = Output_x / 2; ay1 = a; ay2 = Output_y / 2;

thetax = (1/lambda/f) * (sqrt(2*pi)*ax1*ax2*exp(-2*(X1/ax1).^2) + 2*pi*ax2*X1.*erf(sqrt(2)/ax1.*X1));
thetay = (1/lambda/f) * (sqrt(2*pi)*ay1*ay2*exp(-2*(Y1/ay1).^2) + 2*pi*ay2*Y1.*erf(sqrt(2)/ay1.*Y1));
theta = thetax + thetay;
theta_rad = mod(theta, 2*pi);
theta_rad = padding(theta_rad,pixel);

% ---------- Zernike Base Matrices ----------
phaseImage = ones(pixel_Zernike);
[rows, cols] = size(phaseImage);
[xx, yy] = meshgrid(linspace(-1, 1, cols), linspace(1, -1, rows));
[theta2, rho] = cart2pol(xx, yy);

Zernike_n = 4;
array_m = [];
for nn = 0:Zernike_n
    mm = linspace(-nn,nn,nn+1);
    m = [array_m mm];
    array_m = m;
end
array_n = [];
for i = 1:(Zernike_n+1)
    nn_val = ones(1,i)*(i-1);
    n_array = [array_n nn_val];
    array_n = n_array;
end
mm = m;
nn = n_array;

Z_basis = zeros(15, pixel, pixel, 'single');
Pupil_mask = zeros(pixel, pixel, 'single');

for i = 1:length(mm)
    p = zernike_polynomial(nn(i), mm(i), rho, theta2);
    
    % Mask out outside circle
    for r = 1:rows
        for c = 1:cols
            if sqrt((rows/2-r)^2+(cols/2-c)^2) > rows/2
                p(r,c) = 0; 
            end
        end
    end
    
    % padding to 7620
    Z_basis(i, :, :) = single(padding(p, pixel));
end

% Create Pupil Mask
mask_temp = ones(rows, cols);
for r = 1:rows
    for c = 1:cols
        if sqrt((rows/2-r)^2+(cols/2-c)^2) > rows/2
            mask_temp(r,c) = 0; 
        end
    end
end
Pupil_mask = padding(mask_temp, pixel);

% Find crop center
parameter = [f lambda dx];
grating_para = [theta_deg repeat level];
[v, h] = Find_beam_simulation(parameter,grating_para,Beam_size,'First_order');
v = round(v); h = round(h);

Input_beam = single(Input_beam);
Blazed_phi = single(Blazed_phi);
theta_rad = single(theta_rad);

fprintf('Saving tensors to tensors.mat...\n');
save('tensors.mat', 'Input_beam', 'Blazed_phi', 'theta_rad', 'Z_basis', 'Pupil_mask', 'v', 'h', '-v7.3');
fprintf('Done.\n');

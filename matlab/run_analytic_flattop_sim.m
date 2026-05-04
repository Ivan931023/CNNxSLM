% run_analytic_flattop_sim.m
% Automatically change current folder to the script's location
cd(fileparts(mfilename('fullpath')));
clc; clear all; close all;

% ---------- Beam parameter ---------- %
lambda = 447e-6;
f = 300;
Beam_size = 3.45;     % 1/e^2 intensity
Output_x = 0.4;      % The horizental beam size (1/e^2)
Output_y = 0.1;      % The vertical beam size (1/e^2)

% ---------- Phase pattern parameter ---------- %
pixel = 1080;
pixel_Zernike = 750;
range = 150;
CCD_pixel = 2.2e-3;

% ---------- Grating parameter ---------- %
theta_deg = -90;     % Grating rotate angle(Deg)
repeat = 1;          % The number of times the phase is repeated
level = 12; 

% ---------- Theory solution ---------- %
para = [pixel pixel_Zernike f lambda range];
beam_para = [Beam_size Output_x Output_y];
grating_para = [theta_deg repeat level];

% ---------- Zernike coefficient ---------- %
Z = zeros(1, 15);
% Z(2) = 0.05;      % Example: Vertical tilt
Z(6) = 3.0;      % HUGE Astigmatism
Z(8) = 3.0;      % HUGE Coma

fprintf('Running Simulate_flattop...\n');
data_zoomin_analytic = Simulate_flattop(para, beam_para, grating_para, CCD_pixel, Z);
data_zoomin_analytic = data_zoomin_analytic / max(data_zoomin_analytic(:));

% ---------- Plot and Save ---------- %
f1 = figure('Visible', 'off');
imagesc(data_zoomin_analytic);
colormap turbo;
colorbar;
axis image;
title('Distorted Flattop Beam (Astigmatism + Coma)');
xlabel('2.2 \mu m');
ylabel('2.2 \mu m');

fprintf('Saving image...\n');
exportgraphics(f1, 'distorted_flattop_result.png', 'Resolution', 300);
fprintf('Simulation and plotting completed successfully.\n');

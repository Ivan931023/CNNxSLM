function data_zoomin = Simulate_flattop(para,beam_para,grating_para,CCD_pixel,varargin)
    % para = [pixel_pattern pixel_Zernike f lambda range]
    % beam_para = [Beam_size Output_x Output_y]
    % grating_para = [theta_deg repeat level];

    N = round(para(4)*para(3) / (0.008*CCD_pixel),-1); % 0.008 -> SLM pixel size

    pixel = N;         % All pixel
    pixel_pattern = para(1);
    pixel_Zernike = para(2);
    f = para(3);             % Focal length
    lambda = para(4);          % Wave length (mm)
    Zernike_n = 4;
    range = para(5);
    % unit_power = 1300;
    unit_power = 12000;
    
    Beam_size = beam_para(1);        % 1/e^2 intensity
    Output_x = beam_para(2);
    Output_y = beam_para(3);
    
    % --------- SLM plane ---------
    Nx = pixel;
    Ny = pixel;
    
    dx = 8e-3; % (mm)
    dy = 8e-3; % (mm)
    x = -Nx/2*dx : dx : (Nx/2-1)*dx;
    y = -Ny/2*dy : dy : (Ny/2-1)*dy;
    [X, Y] = meshgrid(x, -y);
    
    % --------- Set the Parameter of input beam ---------
    x0 = 0;
    y0 = 0;
    sig_x = Beam_size / 4;
    sig_y = Beam_size / 4;
    
    Input_beam = exp(-((X-x0).^2/2/(sig_x)^2+(Y-y0).^2/2/(sig_y)^2)) ; % Gaussian beam
    Input_beam = Input_beam ./ max(Input_beam(:));
    Input_beam_origin_power = sum(Input_beam,'all');
    ratio = (unit_power * 10000) / Input_beam_origin_power;
    Input_beam = Input_beam * ratio;
    
    % H = Input_beam(pixel/2,:);
    
    % left_index = find(H > max(H)*1/exp(1)^2,1,'first');
    % right_index = find(H > max(H)*1/exp(1)^2,1,'last');
    
    % sprintf('Beam size = %g(mm)',dx*(right_index-left_index))
    % 
    % sprintf('Simulate intensity : %g (unit)',sum(Input_beam,'all')/10000)
    
    % imagesc_turbo(Input_beam)
    % hold on
    % x_center = round(length(Input_beam)/2);
    % y_center = round(length(Input_beam)/2);
    % radius = pixel_Zernike/2;
    % 
    % rectangle('Position', [x_center-radius, y_center-radius, 2*radius, 2*radius], ...
    %           'Curvature', [1, 1], ... 
    %           'EdgeColor', 'r', ...    
    %           'LineWidth', 2);         
    
    % -------- Grating parameter --------%
    theta_deg = grating_para(1);     % Grating rotate angle(Deg)
    theta_blazed = deg2rad(theta_deg); % deg to rad
    max_phase = 255;     % Maximum brazed grating phase
    min_phase = 0;       % Minimum brazed grating phase
    repeat = grating_para(2);          % The number of times the phase is repeated
    level = grating_para(3);          % Phase level
    
    
    % ---- Zernike coefficient ---- %
    if nargin >= 5 && ~isempty(varargin{1})
        Z = varargin{1};
    else
        z1 = 0; z2 = 0; z3 = 0; z4 = 0; z5 = 0;
        z6 = 0; z7 = 0; z8 = 0; z9 = 0; z10 = 0;
        z11 = 0; z12 = 0; z13 = 0; z14 = 0; z15 = 0;
        Z = [z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15];
    end
    
    % ---------- Zernike polynomial ----------
    phaseImage = ones(pixel_Zernike);
    
    [rows, cols] = size(phaseImage);
    [xx, yy] = meshgrid(linspace(-1, 1, cols), linspace(1, -1, rows));
    [theta2 rho] = cart2pol(xx, yy);
    
    ZZ = Zernike_polynomial_superposition(rho, theta2, Z, Zernike_n);
    exp_z = padding(ZZ, pixel); 
    Angle_exp_z = Angle_0_2pi(exp_z);
    
    
    % -------- Analytic solution --------%
    x1 = 8e-3 * (-pixel_pattern/2:pixel_pattern/2-1);
    y1 = 8e-3 * (-pixel_pattern/2:pixel_pattern/2-1);
    [X1, Y1] = meshgrid(x1,y1);
    
    a = Beam_size / (2*sqrt(2));
    ax1 = a;
    ax2 = Output_x / 2;
    ay1 = a;
    ay2 = Output_y / 2;
    
    % a = 1.1;
    % ax1 = a;
    % ay1 = a;
    % ax2 = 0.2;
    % ay2 = 0.05;
    
    grat = Blazed_grating_rotate(pixel_pattern, max_phase, min_phase, level, repeat, theta_blazed);
    grat = padding(grat,pixel);
    Blazed_phi = 2*pi/255 * grat; % [0 255] -> [0 2pi]
    
    thetax = (1/lambda/f) * (sqrt(2*pi)*ax1*ax2*exp(-2*(X1/ax1).^2) + 2*pi*ax2*X1.*erf(sqrt(2)/ax1.*X1));
    thetay = (1/lambda/f) * (sqrt(2*pi)*ay1*ay2*exp(-2*(Y1/ay1).^2) + 2*pi*ay2*Y1.*erf(sqrt(2)/ay1.*Y1));
    theta = thetax + thetay;
    
    theta_rad = mod(theta, 2*pi);
    theta_rad = padding(theta_rad,pixel);
    Total_phase = Blazed_phi + theta_rad + Angle_exp_z;
    phase = mod(Total_phase,2*pi);
    
    
    ratio = 0.99999;
    no_modulate = 0.01;
    
    origin_power = sum(abs(IDFT(Input_beam)).^2,'all');
    
    power_phase = origin_power*(1-no_modulate)*ratio;
    power_amplitude = origin_power*(1-no_modulate)*(1-ratio);
    power_no_modulate = origin_power*no_modulate;
    
    phase_part = IDFT(Input_beam .* exp(1j*phase));
    amplitude_part = IDFT(Input_beam .* (phase/2/pi));
    no_modulate_part = IDFT(Input_beam);
    
    phase_part = phase_part * sqrt(power_phase / sum(abs(phase_part).^2, 'all'));
    amplitude_part = amplitude_part * sqrt(power_amplitude / sum(abs(amplitude_part).^2, 'all'));
    no_modulate_part = no_modulate_part * sqrt(power_no_modulate / sum(abs(no_modulate_part).^2, 'all'));
    
    result = phase_part + amplitude_part + no_modulate_part;
    
    I = abs(result).^2;
    if repeat*level == 32 && pixel == 6350
        data_zoomin = I(2978-range:2978+range,pixel/2-range:pixel/2+range); % 32
    elseif repeat*level == 8 && pixel == 6350
        data_zoomin = I(2380-range:2380+range,pixel/2-range:pixel/2+range); % 8
    elseif repeat*level == 64 && pixel == 7620
        data_zoomin = I(3692-range:3692+range,pixel/2-range:pixel/2+range); % 8
    else
        % ---- Find first order position ---- %
        parameter = [f lambda dx];
        Mode = 'First_order';
        [v, h] = Find_beam_simulation(parameter,grating_para,Beam_size,Mode);
        v = round(v);
        h = round(h);
        data_zoomin = I(v-range:v+range,h-range:h+range);
    end
    
    % figure
    % imagesc(I,[0 1])
    % axis image
    % colormap turbo
    % colorbar
end

% ------ DFT ------
function result = DFT(u)
    result = fftshift(fft2(ifftshift(u)));
end

% ------ IDFT ------
function result = IDFT(u)
    result = fftshift(ifft2(ifftshift(u)));
end
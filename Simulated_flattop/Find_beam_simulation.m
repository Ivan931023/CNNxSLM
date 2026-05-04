function [v, h] = Find_beam_simulation(parameter,grating_para,Beam_size,Mode)
    % parameter = [f lambda dx]
    % grating_para = [theta_deg repeat level]

    f = parameter(1);             % Focal length
    lambda = parameter(2);          % Wave length (mm)
    % --------- SLM plane ---------
    dx = parameter(3); % (mm)
    dy = parameter(3); % (mm)
    N = round(lambda*f / (dx*2.2e-3),-1);
    Nx = N;
    Ny = N;
    x = -Nx/2*dx : dx : (Nx/2-1)*dx;
    y = -Ny/2*dy : dy : (Ny/2-1)*dy;
    [X, Y] = meshgrid(x, -y);
    
    pixel = N;         % All pixel
    pixel_pattern = 1080;
    % range = 150;
    
    % Beam_size = 3;        % 1/e^2 intensity
    
    % --------- Set the Parameter of input beam ---------
    x0 = 0;
    y0 = 0;
    sig_x = Beam_size / 4;
    sig_y = Beam_size / 4;
    
    Input_beam = exp(-((X-x0).^2/2/(sig_x)^2+(Y-y0).^2/2/(sig_y)^2)) ; % Gaussian beam
    
    
    % ---- Find first order ---- %
    if strcmp(Mode,'First_order')
        % theta_deg = -90;     % Grating rotate angle(Deg)
        theta_blazed = deg2rad(grating_para(1)); % deg to rad
        max_phase = 255;     % Maximum brazed grating phase
        min_phase = 0;       % Minimum brazed grating phase
        repeat = grating_para(2);          % The number of times the phase is repeated
        level = grating_para(3); 
    end

    % ---- Find zero order ---- %
    if strcmp(Mode,'Zero_order')
        % theta_deg = -90;     % Grating rotate angle(Deg)
        theta_blazed = deg2rad(grating_para(1)); % deg to rad
        max_phase = 255;     % Maximum brazed grating phase
        min_phase = 255;       % Minimum brazed grating phase
        repeat = grating_para(2);          % The number of times the phase is repeated
        level = grating_para(3); 
    end



    % % -------- Grating parameter --------%
    % theta_deg = -90;     % Grating rotate angle(Deg)
    % theta_blazed = deg2rad(theta_deg); % deg to rad
    % max_phase = 255;     % Maximum brazed grating phase
    % min_phase = 0;       % Minimum brazed grating phase
    % repeat = 1;          % The number of times the phase is repeated
    % level = 32;          % Phase level
    
    grat = Blazed_grating_rotate(pixel_pattern, max_phase, min_phase, level, repeat, theta_blazed);
    grat = padding(grat,pixel);
    Blazed_phi = 2*pi/255 * grat; % [0 255] -> [0 2pi]
    
    Total_phase = Blazed_phi;
    phase = mod(Total_phase,2*pi);
    
    % ---------- Fourier transform ---------- %
    result = IDFT(Input_beam .* exp(1j*phase));
    
    I = abs(result).^2;
    
    [centers, radii] = imfindcircles(I,[7 200]);
    [~, index] = max(radii);
    centers = centers(index,:);
    h = centers(1);
    v = centers(2);
    
    % data_zoomin = I(v-range:v+range,h-range:h+range);
    % imagesc_turbo(I)
    % imagesc_turbo(data_zoomin)
end
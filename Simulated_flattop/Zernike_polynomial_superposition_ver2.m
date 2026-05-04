function Zernike = Zernike_polynomial_superposition_ver2(pixel, Z, Zernike_n, circle)
    % Zernike --> [0 2pi]
    % pixel : Size of phase pattern
    % Z : Zernike coefficient
    
    % --------- Generate index --------- %
    array_m = [];
    for nn = 0:Zernike_n
        mm = linspace(-nn,nn,nn+1);
        m = [array_m mm];
        array_m = m;
    end
    
    array_n = [];
    for i = 1:(Zernike_n+1)
        nn_val = ones(1,i)*(i-1);
        n = [array_n nn_val];
        array_n = n;
    end

    mm = m;
    nn = n;
    % --------------------------------- %

    if length(Z) ~= length(nn)
        sprintf('Error, The size of Z must be 1*%d.',length(nn))
        return
    end

    Zernike = 0;
    count = 0;
    for i = 1:length(mm)
        p = Zernike_polynomial_ver2(nn(i), mm(i), pixel);
        % p = (p.*Z(i) + 1) * pi; % [-1 1] -> [0 2pi]
        p = p .* circle;
        Zernike = Zernike + p*Z(i);
        if Z(i) == 0
            count = count + 1;
        end
    end
    Zernike = Zernike *2*pi;
    Zernike = mod(Zernike,2*pi);

    [xx, yy] = size(Zernike);
    pixel = xx;
    for i = 1:xx
        for j = 1:yy
            if sqrt((pixel/2-i)^2+(pixel/2-j)^2) > pixel/2
                Zernike(i,j) = 0; 
            end
        end
    end
    % ---- For the case where the Zernike coefficients are all zero ----
    % ---- If we do not set it to one, it will affect phase different at the intersection of the pupil ----
    if count == length(nn)
        Zernike = ones(pixel);
    end
end
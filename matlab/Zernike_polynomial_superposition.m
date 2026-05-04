function Zernike = Zernike_polynomial_superposition(rho, theta, Z, n)
    % The size of Z must be 1*15. If not, function will be return.

    array = [];
    for nn = 0:n
        mm = linspace(-nn,nn,nn+1);
        m = [array mm];
        array = m;
    end
    
    array = [];
    for i = 1:(n+1)
        nn_val = ones(1,i)*(i-1);
        n_array = [array nn_val];
        array = n_array;
    end

    mm = m;
    nn = n_array;

    if length(Z) ~= length(nn)
        sprintf('Error, The size of Z must be 1*%d.',length(nn))
        return
    end

    Zernike = 0;
    count = 0;
    for i = 1:length(mm)
        p = zernike_polynomial(nn(i), mm(i), rho, theta);
        % p = (p.*Z(i) + 1) * pi; % [-1 1] -> [0 2pi]
        Zernike = Zernike + p*Z(i);
        if Z(i) == 0
            count = count + 1;
        end
    end
    Zernike = (Zernike + 1) * pi;
    Zernike = exp(1j*Zernike);

    [xx yy] = size(rho);
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
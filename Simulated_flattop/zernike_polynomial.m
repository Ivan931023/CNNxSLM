function Z = zernike_polynomial(n, m, rho, theta)
    if mod(n - abs(m), 2) ~= 0
        Z = zeros(size(rho));
        sprintf('Error')
        return;
    end

    R = zeros(size(rho));
    for k = 0:(n - abs(m)) / 2
        R = R + ((-1)^k * factorial(n - k)) / ...
              (factorial(k) * factorial((n + abs(m)) / 2 - k) * factorial((n - abs(m)) / 2 - k)) * ...
              (rho.^(n - 2 * k));
    end

    if m > 0
        Z = R .* cos(m * theta);
    else
        Z = R .* sin(-m * theta);
    end

    if m == 0
        Z = R;
    end

    if m == 0 && n == 0
        Z = R .* sin(-m * theta);
    end
end
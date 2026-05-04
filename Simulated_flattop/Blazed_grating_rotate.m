function grating = Blazed_grating_rotate(pixel, max, min, levels, repeat, theta)
    x = 1:pixel;
    y = x;
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    [X_rot, Y_rot] = meshgrid(x, y);
    coords = R * [X_rot(:)'; Y_rot(:)'];
    X_rot = reshape(coords(1, :), size(X_rot));
    Y_rot = reshape(coords(2, :), size(Y_rot));
    
    blazed_grating_function = @(x) mod(floor(x / repeat), levels) * (max - min) / (levels - 1) + min;
    grating = blazed_grating_function(X_rot);
end
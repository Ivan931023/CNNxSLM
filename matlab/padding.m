function data = padding(target, size)   
    phase = 0;
    pixel = size;
    image = target;
    l = length(image(1,:));
    k = length(image(:,1));
    data = zeros(pixel) + phase;
    
    for i = 1:pixel
        for j = 1:pixel
            if i > (pixel-l)/2 && i <= (pixel+l)/2 && j > (pixel-k)/2 && j <= (pixel+k)/2
                data(i,j) = image(i-round((pixel-l)/2),j-round((pixel-k)/2));
            end
        end
    end
    if length(target) == size
        data = target;
    end
end
function result = IDFT(u)
    result = fftshift(ifft2(ifftshift(u)));
end
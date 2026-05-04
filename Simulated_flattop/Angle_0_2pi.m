function angle_convert = Angle_0_2pi(Complex)
    data = angle(Complex);
    angle_convert = mod(data + 2*pi, 2*pi);
end
function [ face_points ] = get_facial_coordinates( filename )
    f=importdata(filename);
    face_points=f.data;
end


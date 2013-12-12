
function quality = qualityError(error)
% requirements:
% decreasing function
% quality=+inf  if error  = 0;
% quality = 0.5 if error  = 1;

%paremeters 
thres = 0.005;

%quality =  0.1./(100*error^2)+0.498;
% quality =  1/(200*error);
% 
% quality = quality/4+0.75;

if error<=thres
    quality = 1/(error*1./thres); %(10*progress)^0.5 +1;
else
    quality = 1 - (error-thres);
    %quality = 0.3/((200*error)^0.3)+0.7;
end
quality = max(quality,0);

end

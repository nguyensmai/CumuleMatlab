
function quality = qualityProgress(progress)
% requirements:
% increasing function
% quality=1   if progress = 0;
% quality =+inf if progress = +inf;
if progress>=0
    quality = (10*progress)^0.3 +1;
else
    quality = 2-exp((-50*progress));
end
quality = max(quality,0);
%quality =  -0.1/(min((100*progress)^0.2-1,-10e-10))+0.4;

%quality = quality/2+0.5;

end

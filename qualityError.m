
function quality = qualityError(error)
% requirements:
% decreasing function
% quality=+inf  if error  = 0;
% quality = 0.5 if error  = 1;
%quality =  0.1./(100*error^2)+0.498;
quality =  0.25/(100*error)+0.45;

quality = quality/4+0.75;
end

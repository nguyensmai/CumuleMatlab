function [pred2, mutated] = copyAndMutate(pred1, inputsSet, dimO,probInput)
%
% Author : Nguyen Sao Mai
% nguyensmai@gmail.com
% nguyensmai.free.fr
%    

%parameters
MUTATE_MASK_PROBABILITY = 0.1;
sizeInput = numel(inputsSet);

% initialisation
bitsOut = zeros(1,dimO);
bitsOut(pred1.maskOut) = 1;
bitsInp = zeros(1, numel(inputsSet));
bitsInp(pred1.maskInp) = 1;
method = 'copy ';
mutated = 0;
probInput = probInput/max(probInput);

% mutation
for iInp=1:sizeInput
    if rand()< MUTATE_MASK_PROBABILITY
        if rand()<probInput(iInp)
        bitsInp(iInp) =  1;
        else
            bitsInp(iInp) =  0;
        end
    end
end
% if rand()< MUTATE_MASK_PROBABILITY/2
%     swap1 = randi(sizeInput,1);
%     bitsInp(swap1) = 1-bitsInp(swap1);
%     mutated = 1;
%     method = ['mutate input '];
if rand()<MUTATE_MASK_PROBABILITY
    swap1 = randperm(dimO,2);
    bitsOut(swap1) = bitsOut(swap1(2:-1:1));
    mutated = 1;
    method = ['mutate output '];
end

R1 = pred1.sizeHid1;
R2 = pred1.sizeHid2;
% if rand()<MUTATE_MASK_PROBABILITY    
%     R1 = abs(normrnd(0,pred1.sizeHid1));
% end
% if rand()<MUTATE_MASK_PROBABILITY    
%     R2 = abs(normrnd(0,pred1.sizeHid2));
% end
sizeHid1 = min([pred1.sizeHid1,R1]);
sizeHid2 = min([pred1.sizeHid2,R2]);

[a maskInp] = find(bitsInp==1);
[a maskOut] = find(bitsOut==1);

if isempty(maskInp)
    maskInp = randi(sizeInput,1);
end

%if isempty(maskOut)
if (sum(maskOut)~=1)
   maskOut = randi(dimO, 1); 
end
    

% copy
pred2         = FFN(maskInp, maskOut, R1, R2, inputsSet);
minInputSize  = min([numel(maskInp), numel(pred1.maskInp)]);
minOutputSize = min([numel(maskOut), numel(pred1.maskOut)]);
pred2.w1(1:minInputSize,1:sizeHid1-1)       = pred1.w1(1:minInputSize,1:sizeHid1-1)+0.05-0.1*rand(minInputSize,sizeHid1-1);
pred2.w2(1:sizeHid1,1:sizeHid2-1)       = pred1.w2(1:sizeHid1,1:sizeHid2-1)+ 0.05 -0.1*rand(sizeHid1,sizeHid2-1);
pred2.w3(1:sizeHid2,1:minOutputSize)      = pred1.w3(1:sizeHid2, 1:minOutputSize) + 0.05-0.1*rand(sizeHid2, minOutputSize);
pred2.method  = [method , num2str(pred1.maskInp), ' to ', num2str(pred1.maskOut)];
end

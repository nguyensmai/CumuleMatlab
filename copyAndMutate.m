function [pred2, mutated] = copyAndMutate(pred1, inputsSet, dimO)
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

% mutation
if rand()< MUTATE_MASK_PROBABILITY
    swap1 = randperm(sizeInput,2);
    swap2 = randperm(sizeInput,2);
    bitsInp(swap1) = bitsInp(swap2);
    mutated = 1;
    method = ['mutate input '];
end

if rand()<MUTATE_MASK_PROBABILITY
    swap1 = randperm(dimO,2);
    swap2 = randperm(dimO,2);
    bitsOut(swap1) = bitsOut(swap2);
    mutated = 1;
    method = ['mutate output '];
end
[a maskInp] = find(bitsInp==1);
[a maskOut] = find(bitsOut==1);

if isempty(maskInp)
    maskInp = randi(sizeInput,1);
end

if isempty(maskOut)
   maskOut = randi(dimO, 1); 
end
    

% copy
pred2         = FFN(maskInp, maskOut, pred1.sizeHid1, pred1.sizeHid2, inputsSet);
minInputSize  = min([numel(maskInp), numel(pred1.maskInp)]);
minOutputSize = min([numel(maskOut), numel(pred1.maskOut)]);
pred2.w1(minInputSize,:)       = pred1.w1(minInputSize,:);
pred2.w3(:,minOutputSize)      = pred1.w3(:, minOutputSize);
pred2.method  = [method , num2str(pred1.maskInp), ' to ', num2str(pred1.maskOut)];
end

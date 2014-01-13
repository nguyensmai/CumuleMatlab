classdef FFN
    
% Feed-forward neural network class
% to learn with backpropagation
%
% Author : Nguyen Sao Mai
% nguyensmai@gmail.com
% nguyensmai.free.fr
%    
    properties
        eta      % double. learning rate
        alpha    % double . momentum term
        sizeInp  % integer. size of the input to the neural network
        sizeHid1 % integer. size of the 1st hidden layer
        sizeHid2 % integer. size of the 2nd hiffen layer
        sizeOut  % integer. size of the output of the neural network
        w1       % matrix. weights vector (input -> 1st hiddent layer)
        w2       % matrix. weights vector (1st -> 2nd hiddent layer)
        w3       % matrix. weights vector (2nd hiddent layer -> output)
        dw1Last  % matrix. last change to the weights (used to update the momentum records)
        dw2Last
        dw3Last
        sseRec    % vector. history of the errors
        meanError % double. most recent mean error value
        progress  % double. omost recent progress value
        quality   % double. quality of the predictor
        maskInp   % vector. indicates which sensorimotor variables are inputs
        maskOut   % vector. indicates which sensorimotor variables are outputs
        idFixed   % boolean to indicate if the FFN is will be reused in Cumule or not
        method    % indicates how the FFn has been created (generation or replication)
        probInput % indicates the probabilities that the output depends on the sensorimotor variables
    end
    
    
    methods
        function obj = FFN(inputMask, outputMask,hiddenSize1, hiddenSize2, inputsSet)
            % Contructor of the feed-forward neural network
            % inputMask :  vector. indicates which sensorimotor variables are input
            % outputMask:  vector. indicates which sensorimotor variables
            % are outputs
            % hiddenSize1, hiddenSize2 : integer. size of hidden layers.
            % inputsSet :  vector: sensorimotor variables
            obj.eta   = 0.1;        % Learning rate. Note: eta = 1 is very large.
            obj.alpha = 0.6;    % Momentum term
            % Add a column of 1's to patterns to make a bias node
            obj.sizeInp  = numel(inputMask)+1;
            obj.sizeHid1  = hiddenSize1;
            obj.sizeHid2  = hiddenSize2;
            obj.sizeOut  = numel(outputMask);
            obj.w1       = 0.5*(1-2*rand(obj.sizeInp,obj.sizeHid1-1));
            obj.w2       = 0.5*(1-2*rand(obj.sizeHid1,obj.sizeHid2-1));
            obj.w3       = 0.5*(1-2*rand(obj.sizeHid2,obj.sizeOut));
            obj.dw1Last   = zeros(size(obj.w1));
            obj.dw2Last   = zeros(size(obj.w2));
            obj.dw3Last   = zeros(size(obj.w3));
            obj.sseRec    = [];
            obj.meanError = 10;
            obj.maskInp   = inputMask;
            obj.maskOut   = outputMask;
            obj.idFixed   = -1;
            obj.probInput = 0.4*ones(size(inputsSet));
            obj.quality = 0;
        end %end function constructor
        
        function [predictedOut, hidWithBias1, hidWithBias2]= predict(obj,input)
            winp_into_hid1 = input * obj.w1;  % Pass patterns through weights
            hid_act1 = 1./(1+exp( - winp_into_hid1)); % Sigmoid of weighted input
            
            hidWithBias1 = [ hid_act1 ones(size(hid_act1,1),1) ];    % Add bias node
            whid_into_hid = hidWithBias1 * obj.w2; % Pass hidden acts through weights
            hid_act2 = 1./(1+exp( - whid_into_hid)); % Sigmoid of weighted input
            
            hidWithBias2 = [ hid_act2 ones(size(hid_act2,1),1) ];    % Add bias node
            whid_into_out = hidWithBias2 * obj.w3; % Pass hidden acts through weights
            predictedOut = 1./(1+exp( - whid_into_out)); % Sigmoid of input to output
        end
        
        function output_error = errorInPredictionVec(obj,input, target)
            [predictedOut ]= predict(obj,input);
            error_vect   = target - predictedOut;   % Error matrix
            output_error = trace(error_vect'*error_vect)/obj.sizeOut; % Sum sqr error, matrix style
        end
        
        function output_error = errorInPrediction(obj,input, target)
            [predictedOut ]= predict(obj,input);
            error_vect   = target - predictedOut;   % Error matrix
            output_error = norm(error_vect,2)/(sqrt(obj.sizeOut)); % Sum sqr error, matrix style
        end
        
        function [sse, predictedOut, obj] = bkprop(obj,input,target)
            w1 = obj.w1;
            w2 = obj.w2;
            w3 = obj.w3;
            [predictedOut, hidWithBias1, hidWithBias2]= predict(obj,input);
            deltas_out = target - predictedOut;   % Error matrix
            parfor i=1:size(deltas_out,1)
            ssei(i) = norm(deltas_out(i,:)); % Sum sqr error, matrix style
            end
            sse = mean(ssei)/sqrt(obj.sizeOut); % Sum sqr error, matrix style

            % delta=dE/do * do/dnet
            deltas_hid2 = deltas_out*w3';
            deltas_hid2(:,size(deltas_hid2,2)) = [];
            deltas_hid1 = deltas_hid2*w2';
            deltas_hid1(:,size(deltas_hid1,2)) = [];
            % Take out error signals for bias node
            
            % The key backprop step, in matrix form
            df1 = hidWithBias1 .* (1-hidWithBias1);
            dw1 = obj.eta * input' * (deltas_hid1.* df1(:,1:end-1));
            dw1 = dw1 + obj.alpha * obj.dw1Last; 
            df2 = hidWithBias2 .* (1-hidWithBias2);
            dw2 = obj.eta * hidWithBias1' * (deltas_hid2.*df2(:,1:end-1));
            dw2 = dw2 + obj.alpha * obj.dw2Last;
            df3 = (predictedOut .* (1-predictedOut));
            dw3 = obj.eta * hidWithBias2' * (deltas_out.* df3);
            dw3 = dw3 + obj.alpha * obj.dw3Last;
            obj.w1 = w1 + dw1; obj.w2 = w2 + dw2; obj.w3 = w3 + dw3;     % Weight update
            obj.dw1Last = dw1; obj.dw2Last = dw2; obj.dw3Last = dw3;     % Update momentum records
            obj.sseRec = [obj.sseRec sse];
        end
        
        
       
        
        function [deprecated,  obj] = deprecateBadPredictor( obj, memory, time, timeWindow)
            global tdLearner
            deprecated = false;
            if time>3*timeWindow+1 && numel(obj.sseRec)>3*timeWindow+1
                %obj.meanError  = mean(obj.sseRec(end-timeWindow:end));
                
                current_error = zeros(timeWindow-1,1);
                parfor i=1:timeWindow-1
                    data_in          = memory(i,[obj.maskInp end]);
                    desired_out      = memory(i+1,[obj.maskOut]);
                    current_error(i) = errorInPrediction(obj,data_in, desired_out);
                end
                progressL      = obj.sseRec(end-timeWindow+1:end-1) - current_error';
%                 rew = mean(obj.sseRec(end:-dt:end));
                nbBins = floor(tdLearner.tileC.dimension/2)+1;
                dt = floor((timeWindow-2)/nbBins);
%                 parfor i=1:nbBins
%                     st(i) = mean(obj.sseRec(end-1-nbBins*dt+(i-1)*dt : end-1-nbBins*dt+i*dt));
%                 end
%                 parfor i=1:nbBins
%                     st(nbBins+i) = mean(progressL(end-1-nbBins*dt+(i-1)*dt : end-1-nbBins*dt+(i)*dt));
%                 end
                
                parfor i=1:nbBins
                    meanError(i) = max(10^-10,mean(obj.sseRec(end-nbBins*dt+(i-1)*dt : end-nbBins*dt+i*dt)));
                end
                parfor i=1:nbBins
                    meanProgress(i) = mean(progressL(end-nbBins*dt+(i-1)*dt : end-nbBins*dt+i*dt));
                end
                obj.meanError = meanError(end);
                obj.progress  = meanProgress(end);
                
                if all(meanProgress<-0.01)
                    obj.quality = 10;
                else
                    meanProgress = max(10^-10,meanProgress);

                    st(1:nbBins-1) = meanError(1:nbBins-1);
                    st(nbBins:2*(nbBins-1)) = meanProgress(1:nbBins-1);
                    
                    stp1(1:nbBins-1) = meanError(2:nbBins);
                    stp1(nbBins:2*(nbBins-1)) = meanProgress(2:nbBins);
                    
                    rew = max(10^-10, mean(obj.sseRec(end-dt : end)));
                    
                    %[obj.quality tdLearner, delta] = tdUpdate(tdLearner, -log10(st), -log10(stp1), rew);
                    obj.quality = predict(tdLearner,  -log10(stp1));
                end
                
                if obj.quality>1 + rand() +0.3*exp(-numel(obj.sseRec)/10^4)
                    deprecated = true;
                end
            end
        end %end function deprecateBadPredictors
        
    end %end methods
    
    
end
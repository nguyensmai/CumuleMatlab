classdef FFN
    
    properties
        eta
        alpha
        sizeInp
        sizeHid1
        sizeHid2
        sizeOut
        w1
        w2
        w3
        dw1Last
        dw2Last
        dw3Last
        sseRec
        meanError
        progress
        quality
        maskInp
        maskOut
        idFixed
        method
        probInput
    end
    
    
    methods
        function obj = FFN(inputMask, outputMask,hiddenSize1, hiddenSize2, inputsSet)
            %%%%%%%%%%%%%%%%%%% Initial setting up of the variables
            obj.eta = 0.1;        % Learning rate. Note: eta = 1 is very large.
            obj.alpha = 0.95;    % Momentum term
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
            obj.maskInp   = inputMask;
            obj.maskOut   = outputMask;
            obj.idFixed   = -1;
            obj.probInput = ones(size(inputsSet));
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
        
        function output_error = errorInPrediction(obj,input, target)
            [predictedOut ]= predict(obj,input);
            error_vect   = target - predictedOut;   % Error matrix
            output_error = trace(error_vect'*error_vect)/obj.sizeOut; % Sum sqr error, matrix style
        end
        
        function [sse, predictedOut, obj] = bkprop(obj,input,target)
            w1 = obj.w1;
            w2 = obj.w2;
            w3 = obj.w3;
            dw1Last = obj.dw1Last;
            dw2Last = obj.dw2Last;
            dw3Last = obj.dw3Last;
            [predictedOut hidWithBias1 hidWithBias2]= predict(obj,input);
            output_error = target - predictedOut;   % Error matrix
            sse = trace(output_error'*output_error)/obj.sizeOut; % Sum sqr error, matrix style
            deltas_out = output_error ;
            % delta=dE/do * do/dnet
            deltas_hid2 = deltas_out*w3';
            deltas_hid2(:,size(deltas_hid2,2)) = [];
            deltas_hid1 = deltas_hid2*w2';
            deltas_hid1(:,size(deltas_hid1,2)) = [];
            % Take out error signals for bias node
            
            % The key backprop step, in matrix form
            df1 = hidWithBias1 .* (1-hidWithBias1);
            dw1 = obj.eta * input' * (deltas_hid1.* df1(1:end-1));
            dw1 = dw1 + obj.alpha * dw1Last;
            df2 =   hidWithBias2 .* (1-hidWithBias2);
            dw2 = obj.eta * hidWithBias1' * (deltas_hid2.*df2(1:end-1));
            dw2 = dw2 + obj.alpha * dw2Last;
            dw3 = obj.eta * hidWithBias2' * deltas_out.* predictedOut .* (1-predictedOut);
            dw3 = dw3 + obj.alpha * dw3Last;
            obj.w1 = w1 + dw1; obj.w2 = w2 + dw2; obj.w3 = w3 + dw3;     % Weight update
            obj.dw1Last = dw1; obj.dw2Last = dw2; obj.dw3Last = dw3;     % Update momentum records
            obj.sseRec = [obj.sseRec sse];
        end
        
        
        function [deprecated  obj] = deprecateBadPredictor(obj, memory, time, iTime, timeWindow)
            deprecated = false;
            if time>timeWindow+1 && numel(obj.sseRec)>timeWindow+1
                obj.meanError  = mean(obj.sseRec(end-timeWindow:end));
                qualityPredictor2 = qualityError(obj.meanError) ;
                
                current_error = zeros(timeWindow-1,1);
				for i=timeWindow:-1:iTime+1
                    data_in          = memory(i-iTime,[obj.maskInp end]);
                    desired_out      = memory(i,[obj.maskOut]);
                    current_error(i) = errorInPrediction(obj, data_in, desired_out);
                end
                obj.progress       = obj.meanError - mean(current_error);
                qualityPredictor1     = qualityProgress(obj.progress);
                obj.quality   = (qualityPredictor1*qualityPredictor2)^((numel(obj.sseRec)*1./(20*timeWindow))^2);
                %obj.quality   = exp((qualityPredictor1*qualityPredictor2-1)^(timeWindow^2/numel(obj.sseRec)));
                qualityPredictor      = min(1,obj.quality);
                %qualityPredictor      = min(1,exp((qualityPredictor1*qualityPredictor2-1)^3*timeWindow));
                r = rand()*0.995;
                
                if (r>qualityPredictor) && (obj.idFixed == -1)
                    disp(['deprecate predictor  from ', num2str(obj.maskInp), ' to ', num2str(obj.maskOut),...
                        ' error is ', num2str(obj.meanError), ...
                        ' progress is ', num2str(obj.progress),...
                        ' quality is ', num2str(qualityPredictor2), ' ',num2str(qualityPredictor1), ' ',num2str(obj.quality), ...
                        ' at time ', num2str(numel(obj.sseRec))     ]);
                    if obj.maskOut == 1 || obj.maskOut == 7
                        disp(['DEPRECATEBADPREDICTORS why?', obj.maskOut])
                    end
                    deprecated = true;
                    delta = zeros(1,obj.sizeInp-1);
                    for inp = 1:obj.sizeInp-1
                        delta(inp) = sum(abs(obj.w1(inp,:)));
                    end
                    delta= delta/sum(delta);
                    obj.probInput(obj.maskInp) = max( 0.1,  obj.probInput(obj.maskInp)-0.1*delta); 
                    %                     pred =  pred([1:iPred-1 iPred+1:end]);
                    %                     [pred nPred] = multiplicatePredictors(inputsSet, pred,dimO,dimM, maskOut);
                else
                    %             disp(['good predictor  from ', num2str(obj.maskInp), ' to ', num2str(obj.maskOut),...
                    %                 ' error is ', num2str(obj.meanError), ...
                    %                 ' progress is ', num2str(obj.progress), ...
                    %                 ' quality is ', num2str(qualityPredictor2), ' ',num2str(qualityPredictor1), ' ',num2str(obj.quality), ...
                    %                 ' at time ', num2str(numel(obj.sseRec))     ]);
                    deprecated = false;
                end
                
            end
        end %end function deprecateBadPredictors
        
        end %end methods
        
        
    end
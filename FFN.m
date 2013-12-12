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
        
    end
    
    
    methods
        function obj = FFN(inputMask, outputMask,hiddenSize1, hiddenSize2)
            %%%%%%%%%%%%%%%%%%% Initial setting up of the variables
            obj.eta = 0.01;        % Learning rate. Note: eta = 1 is very large.
            obj.alpha = 0.95;    % Momentum term
            % Add a column of 1's to patterns to make a bias node
            obj.sizeInp  = numel(inputMask)+1;
            obj.sizeHid1  = hiddenSize1;
            obj.sizeHid2  = hiddenSize2;
            obj.sizeOut  = numel(outputMask);
            obj.w1       = 0.5*(1-2*rand(obj.sizeInp,obj.sizeHid1-1));
            obj.w2       = 0.5*(1-2*rand(obj.sizeHid1,obj.sizeHid2-1));
            obj.w3       = 0.5*(1-2*rand(obj.sizeHid2,obj.sizeOut));
            obj.dw1Last = zeros(size(obj.w1));
            obj.dw2Last = zeros(size(obj.w2));
            obj.dw3Last = zeros(size(obj.w3));
            obj.sseRec  = [];
            obj.maskInp = inputMask;
            obj.maskOut = outputMask;
            obj.idFixed = -1;
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
    end
    
    
end
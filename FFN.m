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
        sizeOut  % integer. size of the output of the neural network
        w1       % matrix. weights vector (input -> 1st hiddent layer)
        w2       % matrix. weights vector (1st -> 2nd hiddent layer)
        dw1Last  % matrix. last change to the weights (used to update the momentum records)
        dw2Last
        sseRec    % vector. history of the errors
        meanError % double. most recent mean error value
        progress  % double. omost recent progress value
        quality   % double. quality of the predictor
        maskInp   % vector. indicates which sensorimotor variables are inputs
        maskOut   % vector. indicates which sensorimotor variables are outputs
        maskPruned %pruned out input
        idFixed   % boolean to indicate if the FFN is will be reused in Cumule or not
        method    % indicates how the FFn has been created (generation or replication)
        probInput % indicates the probabilities that the output depends on the sensorimotor variables
        delay     % prediction for time t+delay
        indOutDelay
    end
    
    
    methods
        function obj = FFN(inputMask, outputMask,hiddenSize1, inputsSet,delay)
            % Contructor of the feed-forward neural network
            % inputMask :  vector. indicates which sensorimotor variables are input
            % outputMask:  vector. indicates which sensorimotor variables
            % are outputs
            % hiddenSize1, hiddenSize2 : integer. size of hidden layers.
            % inputsSet :  vector: sensorimotor variables
            obj.eta   = 0.01;        % Learning rate. Note: eta = 1 is very large.
            obj.alpha = 0.1;    % Momentum term
            % Add a column of 1's to patterns to make a bias node
            obj.sizeInp  = numel(inputsSet)+1; %numel(inputMask)+1;
            obj.sizeHid1  = hiddenSize1;
            obj.sizeOut  = numel(outputMask);
            obj.w1       = 0.5*(1-2*rand(obj.sizeInp,obj.sizeHid1-1));
            obj.w2       = 0.5*(1-2*rand(obj.sizeHid1,obj.sizeOut));
            obj.dw1Last   = zeros(size(obj.w1));
            obj.dw2Last   = zeros(size(obj.w2));
            obj.sseRec    = [];
            obj.meanError = 10;
            obj.maskInp   = inputsSet;%inputMask;
            obj.maskOut   = outputMask;
            obj.maskPruned   = [];
            obj.idFixed   = -1;
            obj.probInput = 0.4*ones(size(inputsSet));
            obj.quality = 0;
            obj.delay = 1; %min(max(1,delay),10);
            obj.indOutDelay = (obj.delay-1)*17 +obj.maskOut;
        end %end function constructor
        
        function [predictedOut, hidWithBias]= predict(obj,input)
            winp_into_hid = input * obj.w1;                       % Pass patterns through weights
            hid_act = 1./(1+exp( - winp_into_hid));               % Sigmoid of weighted input
            hidWithBias = [ hid_act ones(size(hid_act,1),1) ];    % Add bias node
            predictedOut = hidWithBias * obj.w2;                % linear transfer to output
        end
        
      
        
        function output_error = errorInPrediction(obj,input, target)
            [predictedOut ]= predict(obj,input);
            error_vect   = target - predictedOut;   % Error matrix
            output_error = trace(error_vect'*error_vect)/obj.sizeOut;  % Sum sqr error, matrix style
        end
        
        function [sse, predictedOut, obj, output_error] = bkprop(obj,input,target)
            [predictedOut, hidWithBias]= predict(obj,input);
            output_error = target - predictedOut; % Error matrix
            sse = trace(output_error'*output_error)/obj.sizeOut;  % Sum sqr error, matrix style
            obj.sseRec = [obj.sseRec sse];                        % Record keeping
            deltas_out = output_error;                            % linear transfer output
            % delta=dE/do * do/dnet
            deltas_hid = deltas_out*obj.w2' .* hidWithBias .* (1-hidWithBias); %size nbOut x sizeHid 
            deltas_hid(:,size(deltas_hid,2)) = [];
            % Take out error signals for bias node
            dw1 = obj.eta * input' * deltas_hid + obj.alpha * obj.dw1Last;     %size sizeInp x sizeHid 
            % The key backprop step, in matrix form
            dw2 = obj.eta * hidWithBias' * deltas_out + obj.alpha * obj.dw2Last;
            obj.w1 = obj.w1 + dw1; obj.w2 = obj.w2 + dw2;           % Weight update
            obj.dw1Last = dw1; obj.dw2Last = dw2;         % Update momentum records
        end
        
         
        
        
        function obj = pruning(obj)
            thresPruning = 0.1;
            max1 = max(abs([obj.w1(:);obj.w2(:);obj.w3(:)]));
            obj.w1=(1-10^-6)*obj.w1;
            obj.w2=(1-10^-6)*obj.w2;
            obj.w3=(1-10^-6)*obj.w3;
            in1= find(mean(abs(obj.w1(1:end-1,:)),2) >thresPruning);
            if ~isempty(in1)
                %                 if ~isempty(setdiff(1:obj.sizeInp-1,in1))
                obj.w1=obj.w1([in1; end],:);
                obj.dw1Last=obj.dw1Last([in1; end],:);
                obj.maskPruned=[obj.maskPruned obj.maskInp(setdiff(1:obj.sizeInp-1,in1)) ];
                %                 end
                obj.maskInp=obj.maskInp(in1);
                obj.sizeInp  = numel(obj.maskInp)+1;
            end
            
            hid1 = find(mean(abs(obj.w1),1) + mean(abs(obj.w2(1:end-1,:)),2)'>thresPruning);
            if ~isempty(hid1)
                obj.w1= obj.w1(:,hid1);
                obj.w2= obj.w2([hid1 end],:);
                obj.dw1Last= obj.dw1Last(:,hid1);
                obj.dw2Last= obj.dw2Last([hid1 end],:);
                obj.sizeHid1  = numel(hid1)+1;
            end
            
            hid2 = find(mean(abs(obj.w2),1) + mean(abs(obj.w3(1:end-1,:)),2)'>thresPruning);
            if ~isempty(hid2)
                obj.w2= obj.w2(:,hid2);
                obj.w3= obj.w3([hid2 end],:);
                obj.dw2Last= obj.dw2Last(:,hid2);
                obj.dw3Last= obj.dw3Last([hid2 end],:);
                obj.sizeHid2  = numel(hid2)+1;
            end
            
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
    end
    
    methods(Static=true)
        function testBackprop()
            %constant function
            pred         = FFN([1], [1], 3, 3, [1], 1);
            for i=1:10000
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[rand() 1],1);
            end
            [predictedOut, ~]= predict(pred,[1 1])    %expects 1
            [predictedOut, ~]= predict(pred,[0 1])    %expects 1
            [predictedOut, ~]= predict(pred,[0.5 1])  %expects 1
            
            
            %linear function
            pred         = FFN([1], [1], 10, 10, [1], 1);
            for i=1:10000
                x= rand();
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[x 1],x);
            end
            [predictedOut, ~]= predict(pred,[1 1])  %expects 1
            [predictedOut, ~]= predict(pred,[0 1])  %expects 0
            [predictedOut, ~]= predict(pred,[0.5 1])%expects 0.5
            
            
            % 2D inputs
            pred         = FFN([1 2], [1], 10, 10, [1 2], 1);
            input = [1  0 1; ...
                0  1 1; ...
                1  1 1; ...
                0  0 1; ...
                0.5 0.5 1]; 
            target = [0.5; 0.5; 1; 0; 0.5];
            test_error = [];
            while true
            for i=1:100
                x= rand();
                y= rand();
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[x y 1],(x+y)/2);
            end
            test = errorInPrediction(pred,input, target);
            test_error= [test_error; test];
            plot(test_error)
            end
            
            [predictedOut(1), ~]= predict(pred,[1  0 1])  %expects 0.5
            [predictedOut(2), ~]= predict(pred,[0  1 1])  %expects 0.5
            [predictedOut(3), ~]= predict(pred,[1  1 1])  %expects 1
            [predictedOut(4), ~]= predict(pred,[0  0 1])  %expects 0
            [predictedOut(5), ~]= predict(pred,[0.5 0.5 1])%expects 0.5
            figure; plot([predictedOut;target']')
        end

         function testBatchBackprop()
           
            % 2D inputs
            pred         = FFN([1 2], [1], 10, 10, [1 2], 1);
            input = [1  0 1; ...
                0  1 1; ...
                1  1 1; ...
                0  0 1; ...
                0.5 0.5 1]; 
            target = [0.5; 0.5; 1; 0; 0.5];
            test_error = [];
            while true
                for i=1:100
                    x(i,1)= rand();
                    y(i,1)= rand();
                end
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[x y ones(100,1)],(x+y)/2);
                
                test = errorInPrediction(pred,input, target);
                test_error= [test_error; test];
                plot(test_error)
            end
            
            [predictedOut(1), ~]= predict(pred,[1  0 1])  %expects 0.5
            [predictedOut(2), ~]= predict(pred,[0  1 1])  %expects 0.5
            [predictedOut(3), ~]= predict(pred,[1  1 1])  %expects 1
            [predictedOut(4), ~]= predict(pred,[0  0 1])  %expects 0
            [predictedOut(5), ~]= predict(pred,[0.5 0.5 1])%expects 0.5
            figure; plot([predictedOut;target']')
        end
        
        function testPruning()
            pred         = FFN([1 2 3 4], [1], 3, 3, [1 2 3 4], 1);
            inpTest = [ [1 1 1 1 ];...
                [0 1 1 1 ];...
                [0.5 1 1 1];...
                [1 0 1 1 ];...
                [0 1 0 1 ];...
                [0.5 0 1 1];...
                [1 1 0 1 ];...
                [0 1 0 1 ];...
                [0.5 1 0 1];...
                [1 1 1 0 ];...
                [0 1 1 0 ];...
                [0.5 1 1 0];...
                [1 1 0 0 ];...
                [0 1 0 0 ];...
                [0.5 0 1 0];];
            targetTest = [
            for i=1:10000
                x= rand();
                inp = [x rand() rand() rand()];
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[inp(pred.maskInp) 1],x);
                pred = pruning(pred);
                test = errorInPrediction(pred,inpTest, target);
                test_error= [test_error; test];
                plot(test_error)
            end

            [predictedOut, ~]= predict(pred,[inp(1,pred.maskInp) 1])  %expects 1
            [predictedOut, ~]= predict(pred,[inp(2,pred.maskInp) 1])  %expects 0
            [predictedOut, ~]= predict(pred,[inp(3,pred.maskInp) 1])  %expects 0.5
            [predictedOut, ~]= predict(pred,[inp(4,pred.maskInp) 1])  %expects 1
            [predictedOut, ~]= predict(pred,[inp(5,pred.maskInp) 1])  %expects 0
            [predictedOut, ~]= predict(pred,[inp(6,pred.maskInp) 1])  %expects 0.5
            [predictedOut, ~]= predict(pred,[inp(7,pred.maskInp) 1])  %expects 1
            [predictedOut, ~]= predict(pred,[inp(8,pred.maskInp) 1])  %expects 0
            [predictedOut, ~]= predict(pred,[inp(9,pred.maskInp) 1])  %expects 0.5
            
            
            
            pred         = FFN([1 2 3 4], [1], 10, 10, [1 2 3 4], 1);
            inp = [ [1 1 1 1 ];...
                [0 1 1 1 ];...
                [0.5 1 1 1];...
                [1 0 1 1 ];...
                [0 1 0 1 ];...
                [0.5 0 1 1];...
                [1 1 0 1 ];...
                [0 1 0 1 ];...
                [0.5 1 0 1];...
%                 [1 1 1 0 ];...
%                 [0 1 1 0 ];...
%                 [0.5 1 1 0];...
%                 [1 1 0 0 ];...
%                 [0 1 0 0 ];...
%                 [0.5 0 1 0];...
                ];
            target= [1;0.5;0.75;1;0;0.25;0.5;0;0.25];
            for i=1:1000
                x= rand();
                y= rand();
                inp = [x rand() y rand()];
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[inp(pred.maskInp) 1],...
                    (x+y)/2);
                % pred = pruning(pred);
            end
            
            [predictedOut, ~]= predict(pred,[inp(1,pred.maskInp) 1])  %expects 1
            [predictedOut, ~]= predict(pred,[inp(2,pred.maskInp) 1])  %expects 0.5
            [predictedOut, ~]= predict(pred,[inp(3,pred.maskInp) 1])  %expects 0.75
            [predictedOut, ~]= predict(pred,[inp(4,pred.maskInp) 1])  %expects 1
            [predictedOut, ~]= predict(pred,[inp(5,pred.maskInp) 1])  %expects 0
            [predictedOut, ~]= predict(pred,[inp(6,pred.maskInp) 1])  %expects 0.25
            [predictedOut, ~]= predict(pred,[inp(7,pred.maskInp) 1])  %expects 0.5
            [predictedOut, ~]= predict(pred,[inp(8,pred.maskInp) 1])  %expects 0
            [predictedOut, ~]= predict(pred,[inp(9,pred.maskInp) 1])  %expects 0.25
            
        end
    end %end methods
    
    
end
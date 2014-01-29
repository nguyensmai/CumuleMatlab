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
        nbHid    % integer. number of hidden layers
        sizeHid % list of integer. size of the 1st hidden layer
        sizeOut  % integer. size of the output of the neural network
        w       % matrix. weights vector (input -> 1st hiddent layer)
        wOut       % matrix. weights vector (1st -> 2nd hiddent layer)
        dwLast  % matrix. last change to the weights (used to update the momentum records)
        dwOutLast
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
        
        %{
        structure of the ffn
        input (sizeInput) --- w{1} (sizeInp x (sizeHid(1)-1) )
        --> winp_into_hid{1}  --- transfer function -->
        hid_act{1} --- w{2} (sizeHid(1)x(sizeHid(2)-1)) --> winp_into_hid{2}
        ...
        hid_act{nbHid-1} --- w{nbHid} (sizeHid(nbHid-1)x(sizeHid(nbHid)-1))
        --> winp_into_hid{nbHid} ---transfer function -->
        hid_act{nbHid} --- wOut (sizeHid(nbHid)x(sizeOut)) --> predicted_out
        %}
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
            obj.sizeInp  = numel(inputMask)+1;
            obj.sizeHid  = hiddenSize1;
            obj.nbHid    = numel(obj.sizeHid);
            obj.sizeOut  = numel(outputMask);
            obj.w{1}       = 0.5*(1-2*rand(obj.sizeInp,obj.sizeHid(1)-1));
            obj.dwLast{1}   = zeros(size(obj.w{1}));
            for iHid =2:obj.nbHid
                obj.w{iHid}       = 0.5*(1-2*rand(obj.sizeHid(iHid-1),obj.sizeHid(iHid)-1));
                obj.dwLast{iHid}   = zeros(size(obj.w{iHid}));
            end
            obj.wOut       = 0.5*(1-2*rand(obj.sizeHid(end),obj.sizeOut));
            obj.dwOutLast   = zeros(size(obj.wOut));
            obj.sseRec    = [];
            obj.meanError = 10;
            obj.maskInp   = inputMask;
            obj.maskOut   = outputMask;
            obj.maskPruned   = [];
            obj.idFixed   = -1;
            obj.probInput = 0.4*ones(size(inputsSet));
            obj.quality = 0;
            obj.delay = 1; %min(max(1,delay),10);
            obj.indOutDelay = (obj.delay-1)*17 +obj.maskOut;
        end %end function constructor
        
        function [predictedOut, hidWithBias]= predict(obj,input)
            %initalise
            winp_into_hid = cell(obj.nbHid);
            hid_act       = cell(obj.nbHid);
            %feeed forward input -> hidden1
            winp_into_hid{1} = input * obj.w{1};                       % Pass patterns through weights
            hid_act{1} = 1./(1+exp( - winp_into_hid{1}));               % Sigmoid of weighted input
            hidWithBias{1} = [ hid_act{1} ones(size(hid_act{1},1),1) ];    % Add bias node
            for iHid =2:obj.nbHid
                winp_into_hid{iHid} = hidWithBias{iHid-1} * obj.w{iHid};                       % Pass patterns through weights
                hid_act{iHid} = 1./(1+exp( - winp_into_hid{iHid}));               % Sigmoid of weighted input
                hidWithBias{iHid} = [ hid_act{iHid} ones(size(hid_act{iHid},1),1) ];    % Add bias node
            end
            predictedOut = hidWithBias{end} * obj.wOut;                % linear transfer to output
        end
        
      
        % output_error is the matrix of square error for each dim and data 
        function output_error = errorInPrediction(obj,input, target, plotB)
            [predictedOut ]= predict(obj,input);
            error_vect   = target - predictedOut;   % Error matrix
            output_error = trace(error_vect'*error_vect)/obj.sizeOut;  % Sum sqr error, matrix style
            %output_error= output_error.*output_error;
            if exist('plotB','var')
                    if  obj.sizeOut==1
                        plot([target(:) predictedOut(:)])
                    else
                        for iOut =plotB
                            subplot(1,numel(plotB),iOut)
                            plot([target(:,iOut) predictedOut(:,iOut)])
                        end
                    end
            end
        end
        
        function [sse, predictedOut, obj, output_error] = bkprop(obj,input,target)
            %{
            backprop structure
            deltas_out --- wOut --> 
            deltas_hid{nbHid} --- w{nbHid} -->
            deltas_hid{nbHid-1} --- w{nbHid-1} --> ...
            deltas_hid{2} --- w{2} --> deltas_hid{1} --- w{1} --> input
            %}
            [predictedOut, hidWithBias]= predict(obj,input);
            output_error = target - predictedOut; % Error matrix
            sse = trace(output_error'*output_error)/(obj.sizeOut * size(input,1));  % Sum sqr error, matrix style
            obj.sseRec = [obj.sseRec sse];                        % Record keeping
            deltas_out = output_error;                            % linear transfer output
            % delta=dE/do * do/dnet
            deltas_hid{obj.nbHid} = deltas_out*obj.wOut' .* hidWithBias{obj.nbHid} .* (1-hidWithBias{obj.nbHid}); %size nbOut x sizeHid
            deltas_hid{obj.nbHid}(:,size(deltas_hid{obj.nbHid},2)) = [];
            for iHid =obj.nbHid-1:-1:1
                deltas_hid{iHid} = deltas_hid{iHid+1}*obj.w{iHid+1}' .* hidWithBias{iHid} .* (1-hidWithBias{iHid}); %size nbOut x sizeHid
                deltas_hid{iHid}(:,size(deltas_hid{iHid},2)) = [];
            end
            
            % Take out error signals for bias node
            dw{1} = obj.eta * input' * deltas_hid{1} + obj.alpha * obj.dwLast{1};     %size sizeInp x sizeHid
            obj.w{1} = obj.w{1} + dw{1};
            obj.dwLast{1} = dw{1};
            for iHid =2:obj.nbHid
                dw{iHid} = obj.eta * hidWithBias{iHid-1}' * deltas_hid{iHid} + obj.alpha * obj.dwLast{iHid};     %size sizeInp x sizeHid
                obj.w{iHid} = obj.w{iHid} + dw{iHid};
                obj.dwLast{iHid} = dw{iHid};
            end
            % The key backprop step, in matrix form
            dwOut = obj.eta * hidWithBias{end}' * deltas_out + obj.alpha * obj.dwOutLast;
             obj.wOut = obj.wOut + dwOut;           % Weight update
             obj.dwOutLast = dwOut;         % Update momentum records
        end
        
         
        
        
        function obj = pruning(obj)
            thresPruning = 0.1;
            max1 = max(abs([obj.w(:);obj.wOut(:)]));
            obj.w=(1-10^-5)*obj.w;
            obj.wOut=(1-10^-5)*obj.wOut;
            if mean(sum(abs(obj.dwLast),2))<0.1
                in1= find(mean(abs(obj.w(1:end-1,:)),2) >thresPruning*max1);
                if ~isempty(in1)
                    %                 if ~isempty(setdiff(1:obj.sizeInp-1,in1))
                    obj.w=obj.w([in1; end],:);
                    obj.dwLast=obj.dwLast([in1; end],:);
                    obj.maskPruned=[obj.maskPruned obj.maskInp(setdiff(1:obj.sizeInp-1,in1)) ];
                    %                 end
                    obj.maskInp=obj.maskInp(in1);
                    obj.sizeInp  = numel(obj.maskInp)+1;
                end
            end
            if mean(sum(abs(obj.dwOutLast),2))<0.1
                hid1 = find(mean(abs(obj.w),1) + mean(abs(obj.wOut(1:end-1,:)),2)'>2*thresPruning*max1);
                if ~isempty(hid1)
                    obj.w= obj.w(:,hid1);
                    obj.wOut= obj.wOut([hid1 end],:);
                    obj.dwLast= obj.dwLast(:,hid1);
                    obj.dwOutLast= obj.dwOutLast([hid1 end],:);
                    obj.sizeHid  = numel(hid1)+1;
                end
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
            pred         = FFN([1], [1], 3, [1], 1);
            for i=1:10000
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[rand() 1],1);
            end
            [predictedOut, ~]= predict(pred,[1 1])    %expects 1
            [predictedOut, ~]= predict(pred,[0 1])    %expects 1
            [predictedOut, ~]= predict(pred,[0.5 1])  %expects 1
            
            
            %linear function
            pred         = FFN([1], [1], 10, [1], 1);
            for i=1:10000
                x= rand();
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[x 1],x);
            end
            [predictedOut, ~]= predict(pred,[1 1])  %expects 1
            [predictedOut, ~]= predict(pred,[0 1])  %expects 0
            [predictedOut, ~]= predict(pred,[0.5 1])%expects 0.5
            
            
            % 2D inputs
            pred         = FFN([1 2], [1], 10, [1 2], 1);
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
            predictedOut=[]
            [predictedOut(1), ~]= predict(pred,[1  0 1])  %expects 0.5
            [predictedOut(2), ~]= predict(pred,[0  1 1])  %expects 0.5
            [predictedOut(3), ~]= predict(pred,[1  1 1])  %expects 1
            [predictedOut(4), ~]= predict(pred,[0  0 1])  %expects 0
            [predictedOut(5), ~]= predict(pred,[0.5 0.5 1])%expects 0.5
            figure; plot([predictedOut;target']')
            
            
            % 2D inputs, 2D outputs
            pred         = FFN([1 2], [1 2], 10, [1 2], 1);
            input = [1  0 1; ...
                0  1 1; ...
                1  1 1; ...
                0  0 1; ...
                0.5 0.5 1]; 
            target = input(:,1:2)/2;
            test_error = [];
            while true
            for i=1:100
                x= rand();
                y= rand();
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[x y 1],[x y]/2);
            end
            test = errorInPrediction(pred,input, target);
            test_error= [test_error; test];
            plot(test_error)
            end
            predictedOut=[]
            [predictedOut(1,:), ~]= predict(pred,[1  0 1])  %expects 0.5
            [predictedOut(2,:), ~]= predict(pred,[0  1 1])  %expects 0.5
            [predictedOut(3,:), ~]= predict(pred,[1  1 1])  %expects 1
            [predictedOut(4,:), ~]= predict(pred,[0  0 1])  %expects 0
            [predictedOut(5,:), ~]= predict(pred,[0.5 0.5 1])%expects 0.5
            figure; plot([predictedOut;target']')
            
            
 
            % 2 layer- network, 2D inputs
            pred         = FFN([1 2], [1], [5 5], [1 2], 1);
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
            predictedOut=[]
            [predictedOut(1), ~]= predict(pred,[1  0 1])  %expects 0.5
            [predictedOut(2), ~]= predict(pred,[0  1 1])  %expects 0.5
            [predictedOut(3), ~]= predict(pred,[1  1 1])  %expects 1
            [predictedOut(4), ~]= predict(pred,[0  0 1])  %expects 0
            [predictedOut(5), ~]= predict(pred,[0.5 0.5 1])%expects 0.5
            figure; plot([predictedOut;target']')
        end

         function testBatchBackprop()
           
            % 2D inputs
            pred         = FFN([1 2], [1], 10, [1 2], 1);
            inputTest = [1  0 1; ...
                0  1 1; ...
                1  1 1; ...
                0  0 1; ...
                0.5 0.5 1]; 
            target = [0.5; 0.5; 1; 0; 0.5];
            test_error = [];
            while true
                x= rand(100,1);
                y= rand(100,1);
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[x y ones(100,1)],(x+y)/2);
                
                test = errorInPrediction(pred,inputTest, target);
                test_error= [test_error; test];
                semilogy(test_error)
            end
            predictedOut =[]
            [predictedOut(1), ~]= predict(pred,[1  0 1]);  %expects 0.5
            [predictedOut(2), ~]= predict(pred,[0  1 1]);  %expects 0.5
            [predictedOut(3), ~]= predict(pred,[1  1 1]);  %expects 1
            [predictedOut(4), ~]= predict(pred,[0  0 1]);  %expects 0
            [predictedOut(5), ~]= predict(pred,[0.5 0.5 1]);%expects 0.5
            figure; plot([predictedOut;target(1:5)']')
            
            
            % 2D inputs, 3D outputs
            pred         = FFN([1 2], [1 2 3], 10, [1 2 3], 1);
            inputTest = [1  0 1; ...
                0  1 1; ...
                1  1 1; ...
                0  0 1; ...
                0.5 0.5 1]; 
            target = [inputTest(:,1:2) mean(inputTest(:,1:2),2)];
            test_error = [];
            errorLt = [];
            while true
                for i=1:100
                    x(i,1)= rand();
                    y(i,1)= rand();
                end
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[x y ones(100,1)],[x y (x+y)/2]);
                
                test = errorInPrediction(pred,inputTest, target);
                test_error= [test_error; test];
                errorLt = [errorLt; deltas_out.*deltas_out];
                semilogy(test_error)
            end
            figure
            for i=1:3
                subplot(1,3,i)
                semilogy(smooth(abs(errorLt(:,i)),1000))
            end
            predictedOut =[]
            [predictedOut(1,:), ~]= predict(pred,[1  0 1]);  %expects 0.5
            [predictedOut(2), ~]= predict(pred,[0  1 1]);  %expects 0.5
            [predictedOut(3), ~]= predict(pred,[1  1 1]);  %expects 1
            [predictedOut(4), ~]= predict(pred,[0  0 1]);  %expects 0
            [predictedOut(5), ~]= predict(pred,[0.5 0.5 1]);%expects 0.5
            figure; plot([predictedOut;target(1:5)']')
            
            
            
             % 2 layer- network, 2D inputs
            pred         = FFN([1 2], [1], [5 5], [1 2], 1);
            input = [1  0 1; ...
                0  1 1; ...
                1  1 1; ...
                0  0 1; ...
                0.5 0.5 1]; 
            target = [0.5; 0.5; 1; 0; 0.5];
            test_error = [];
            while true
            for i=1:100
                x(i,:)= rand();
                y(i,:)= rand();
            end
            [sse, predictedOut, pred, deltas_out] = bkprop(pred,[x y ones(100,1)],(x+y)/2);
            test = errorInPrediction(pred,input, target);
            test_error= [test_error; test];
            plot(test_error)
            end
            predictedOut=[]
            [predictedOut(1), ~]= predict(pred,[1  0 1])  %expects 0.5
            [predictedOut(2), ~]= predict(pred,[0  1 1])  %expects 0.5
            [predictedOut(3), ~]= predict(pred,[1  1 1])  %expects 1
            [predictedOut(4), ~]= predict(pred,[0  0 1])  %expects 0
            [predictedOut(5), ~]= predict(pred,[0.5 0.5 1])%expects 0.5
            figure; plot([predictedOut;target']')
            
            
            
             % 1D input, 1D output, 2 layers 
            pred         = FFN([1], [1], [5 5], [1], 1);
            inputTest = [sort(rand(10,1)) ones(10,1)];
            target = cos(5*pi*inputTest(:,1));
            test_error = [];
            while true
                x= rand(100,1);
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[x ones(100,1)],cos(x*5*pi));
                
                test = errorInPrediction(pred,inputTest, target);
                test_error= [test_error; test];
                semilogy(test_error)
            end
            predictedOut =[]
            [predictedOut(1), ~]= predict(pred,inputTest(1,:));  %expects 0.5
            [predictedOut(2), ~]= predict(pred,inputTest(2,:));  %expects 0.5
            [predictedOut(3), ~]= predict(pred,inputTest(3,:));  %expects 1
            [predictedOut(4), ~]= predict(pred,inputTest(4,:));  %expects 0
            [predictedOut(5), ~]= predict(pred,inputTest(5,:));%expects 0.5
            figure; plot([predictedOut;target(1:5)']')
            
            
         end
        
        
         function testInputMask()
             
             % 2D inputs
             pred1         = FFN([1 2], [1], 10, [1 2 3], 1);
             pred2         = FFN([3], [1], 10, [1 2 3], 1);
             inputTest = [1  0 0.4 1; ...
                 0  1  0.6 1; ...
                 1  1 1 1; ...
                 0  0 0 1; ...
                 0.5 0.5 0.1  1];
             target = [0.5; 0.5; 1; 0; 0.5];
             test_error = [];
             while true
                 x= rand(100,1);
                 y= rand(100,1);
                 z= rand(100,1);
                 [sse1, predictedOut, pred1, deltas_out] = bkprop(pred1,[x y ones(100,1)],(x+y)/2);
                 [sse2, predictedOut, pred2, deltas_out] = bkprop(pred2,[z ones(100,1)],(x+y)/2);
                 
                 test1 = errorInPrediction(pred1,inputTest(:,[pred1.maskInp end]), target);
                 test2 = errorInPrediction(pred2,inputTest(:,[pred2.maskInp end]), target);
                 test_error= [test_error; test1 test2];
                 semilogy(test_error)
             end
             predictedOut =[]
             [predictedOut(1), ~]= predict(pred,[1  0 1]);  %expects 0.5
             [predictedOut(2), ~]= predict(pred,[0  1 1]);  %expects 0.5
             [predictedOut(3), ~]= predict(pred,[1  1 1]);  %expects 1
             [predictedOut(4), ~]= predict(pred,[0  0 1]);  %expects 0
             [predictedOut(5), ~]= predict(pred,[0.5 0.5 1]);%expects 0.5
             figure; plot([predictedOut;target(1:5)']')
             
         end
             
             
        function testPruning()
            pred         = FFN([1 2 3 4], [1], 3, [1 2 3 4], 1);
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
            inpTest = [inpTest ones(size(inpTest,1),1)];
            targetTest = inpTest(:,1);
            test_error=[];
            for i=1:10000
                for j=1:100
                    x(j,1)= rand();
                end
                inp = [x rand(100,3)];
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[inp(:,pred.maskInp) ones(100,1)],x);
                pred = pruning(pred);
                test = errorInPrediction(pred,inpTest(:,[pred.maskInp end]), targetTest);
                test_error= [test_error; test];
                semilogy(test_error)
                %plot(test_error)
            end

            predictedOut = [];
            [predictedOut(1), ~]= predict(pred,[inpTest(1,pred.maskInp) 1])  %expects 1
            [predictedOut(2), ~]= predict(pred,[inpTest(2,pred.maskInp) 1])  %expects 0
            [predictedOut(3), ~]= predict(pred,[inpTest(3,pred.maskInp) 1])  %expects 0.5
            [predictedOut(4), ~]= predict(pred,[inpTest(4,pred.maskInp) 1])  %expects 1
            [predictedOut(5), ~]= predict(pred,[inpTest(5,pred.maskInp) 1])  %expects 0
            [predictedOut(6), ~]= predict(pred,[inpTest(6,pred.maskInp) 1])  %expects 0.5
            [predictedOut(7), ~]= predict(pred,[inpTest(7,pred.maskInp) 1])  %expects 1
            [predictedOut(8), ~]= predict(pred,[inpTest(8,pred.maskInp) 1])  %expects 0
            [predictedOut(9), ~]= predict(pred,[inpTest(9,pred.maskInp) 1])  %expects 0.5
            figure; plot([predictedOut;targetTest([1:9])']')

            
            
            pred         = FFN([1:13], [1], 10, [1:13], 1);
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
                [0.5 0 1 0];...
                ];
            inpTest = [inpTest rand(size(inpTest,1),9) ones(size(inpTest,1),1)];
            targetTest= mean(inpTest(:,[1 3]),2);
            test_error=[];
            batch = 100;
            for i=1:30000
                x= rand(batch,1);
                y= rand(batch,1);
                inp = [x rand(batch,1) y rand(batch,10)];
                [sse, predictedOut, pred, deltas_out] = bkprop(pred,[inp(:,pred.maskInp) ones(batch,1)],...
                    (x+y)/2);
                %pred = pruning(pred);
                test = errorInPrediction(pred,inpTest(:,[pred.maskInp end]), targetTest);
                test_error= [test_error; test];
                semilogy(test_error)
                %plot(test_error)
            end
            
            predictedOut=[];
            [predictedOut(1), ~]= predict(pred,[inpTest(1,pred.maskInp) 1])  %expects 1
            [predictedOut(2), ~]= predict(pred,[inpTest(2,pred.maskInp) 1])  %expects 0.5
            [predictedOut(3), ~]= predict(pred,[inpTest(3,pred.maskInp) 1])  %expects 0.75
            [predictedOut(4), ~]= predict(pred,[inpTest(4,pred.maskInp) 1])  %expects 1
            [predictedOut(5), ~]= predict(pred,[inpTest(5,pred.maskInp) 1])  %expects 0
            [predictedOut(6), ~]= predict(pred,[inpTest(6,pred.maskInp) 1])  %expects 0.25
            [predictedOut(7), ~]= predict(pred,[inpTest(7,pred.maskInp) 1])  %expects 0.5
            [predictedOut(8), ~]= predict(pred,[inpTest(8,pred.maskInp) 1])  %expects 0
            [predictedOut(9), ~]= predict(pred,[inpTest(9,pred.maskInp) 1])  %expects 0.25
            figure; plot([predictedOut;targetTest([1:9])']')
  
        end
    end %end methods
    
    
end
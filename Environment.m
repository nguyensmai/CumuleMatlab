classdef Environment < handle
    
    properties
        hiddenState;
        dimM;
        dimO;
    end
    
    
    
    methods
        
        
        function obj = Environment(dimO,dimM)
            obj.hiddenState = 0;
            obj.dimO = dimO;
            obj.dimM = dimM;
            obj.hiddenState= [1, 0,0,0,0];

        end
        
        function mt = randomAction(obj)
            mt = rand(1,obj.dimM);
        end

        
        function outputState = executeAction(obj,state, action)
            
            if size(state,2) ~=50
                error(['ENVIRONMENT: error in input state ', num2str(size(state,2)), ' instead of 50']);
            end
            
            if size(action,2) ~=2
                error(['ENVIRONMENT: error in input action', num2str(size(action,2)), ' instead of 2']);
            end
            
            outputState(1) =  cos(action(1)*3*pi);
            outputState(2) =  (state(1)+state(2))/2;
            outputState(3) =  cos(state(1)*pi);
            outputState(4) =  cos(action(1)*pi);
            outputState(5) =  (action(1)+action(2))/2;
            outputState(6) =  cos(action(1)*3*pi);
            outputState(7) =  (state(1)+state(2))/2;
            outputState(8) =  cos(state(1)*pi);
            outputState(9) =  cos(action(1)*pi);
            outputState(10) =  (action(1)+action(2))/2;
            outputState(11) =  cos(action(1)*3*pi);
            outputState(12) =  (state(1)+state(2))/2;
            outputState(13) =  cos(state(1)*pi);
            outputState(14) =  cos(action(1)*pi);
            outputState(15) =  (action(1)+action(2))/2;
            outputState(16) =  cos(action(1)*3*pi);
            outputState(17) =  (state(1)+state(2))/2;
            outputState(18) =  cos(state(1)*pi);
            outputState(19) =  cos(action(1)*pi);
            outputState(20) =  (action(1)+action(2))/2;
            outputState(21) =  cos(action(1)*3*pi);
            outputState(22) =  (state(1)+state(2))/2;
            outputState(23) =  cos(state(1)*pi);
            outputState(24) =  cos(action(1)*pi);
            outputState(25) =  (action(1)+action(2))/2;
            outputState(26) =  cos(action(1)*3*pi);
            outputState(27) =  (state(1)+state(2))/2;
            outputState(28) =  cos(state(1)*pi);
            outputState(29) =  cos(action(1)*pi);
            outputState(30) =  (action(1)+action(2))/2;
            outputState(31) =  cos(action(1)*3*pi);
            outputState(32) =  (state(1)+state(2))/2;
            outputState(33) =  cos(state(1)*pi);
            outputState(34) =  cos(action(1)*pi);
            outputState(35) =  (action(1)+action(2))/2;
            outputState(36) =  cos(action(1)*3*pi);
            outputState(37) =  (state(1)+state(2))/2;
            outputState(38) =  cos(state(1)*pi);
            outputState(39) =  cos(action(1)*pi);
            outputState(40) =  (action(1)+action(2))/2;
            outputState(41) =  cos(action(1)*3*pi);
            outputState(42) =  (state(1)+state(2))/2;
            outputState(43) =  cos(state(1)*pi);
            outputState(44) =  cos(action(1)*pi);
            outputState(45) =  (action(1)+action(2))/2;
            outputState(46) =  cos(action(1)*3*pi);
            outputState(47) =  (state(1)+state(2))/2;
            outputState(48) =  cos(state(1)*pi);
            outputState(49) =  cos(action(1)*pi);
            outputState(50) =  (action(1)+action(2))/2;
            
            %             outputState(2) = min(1, max((state(2)+action(2))/2,-1));
%             outputState(3) = min(1, max(cos((state(3)^2 + state(4)^2+ state(5)^2)+(action(1)^2 + state(4)^2 +state(5)^2)),-1));
%             outputState(4) = min(1, max(cos(state(3)+state(2)),-1)); %sin(obj.hiddenState);
%             outputState(5) = min(1, max(cos(action(1)+action(2)),-1)); %sin(obj.hiddenState);
%             
%             outputState(6) =  min(1, max(0.2*state(3)^2 + state(4)^2- 0.2*state(5)^2,-1)); 
%             outputState(7) =  min(1, max(0.4*state(4)^2 + 4*state(2)^2 +0.4*state(3)^2,-1)); 
%             outputState(8) =  min(1, max(-1 + 0.4*action(1)^2+ 0.4*state(4)^2 +4*state(5)^2,-1)); 
%             outputState(9) = min(1, max((state(9)+action(1))/2,-1));
%             outputState(10) = min(1, max(0.5*state(4)+ 0.5*state(2),-1));%2*rand() -1; 
%             outputState(11) = min(1, max(0.2*state(3)^2 - 0.5*state(4)^2 +0.3* state(5)^2,-1));
%             outputState(12) = min(1, max(0.2*state(4)^2 + 0.4*state(2)^2 -0.8*state(3)^2,-1)); 
%             outputState(13) = min(1, max(cos(2*state(4)+0.4*action(1)),-1));
%             outputState(14) = 1; %mod(obj.hiddenState(1),10)/5-1; 
%             outputState(15) = 2*(2*sigmoid(state(6))-1);
%             outputState(16) = 2*sigmoid(state(14))-1;
%             outputState(17) = 1; %2*sigmoid(obj.hiddenState(2))-1;

%             if state(1)+action(1)<0
%                 outputState(5) = 0;
%             elseif state(1)+action(1)<0.8
%                 outputState(5) = state(1)+action(1);
%             else
%                 outputState(5) = state(2)+action(2);
%             end
            obj.hiddenState= [obj.hiddenState(1)+1, obj.hiddenState(3),obj.hiddenState(4),obj.hiddenState(5), state(4)];
        end %end executeAction
        
        
        function plotMemory(obj,sMemory)
            figure
            subplot(1,obj.dimO,1)
            scatter(sMemory(1:end-1,51), sMemory(2:end,1))
            title('outputState(1) =  cos(action(1)*3*pi)')
            subplot(1,obj.dimO,2)
            scatter((sMemory(1:end-1,1)+sMemory(1:end-1,2))/2, sMemory(2:end,2))
            title('outputState(2) =  (state(1)+state(2))/2')
            subplot(1,obj.dimO,3)
            scatter(sMemory(1:end-1,1), sMemory(2:end,3))
            title('outputState(3) =  cos(state(1)*pi)')

        end

        
        
        
    end %end methods
    
    
end

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
            mt = 2*rand(1,obj.dimM)-1;
        end

        
        function outputState = executeAction(obj,state, action)
            
            if size(state,2) ~=5
                error(['ENVIRONMENT: error in input state ', num2str(size(state,2)), ' instead of 5']);
            end
            
            if size(action,2) ~=2
                error(['ENVIRONMENT: error in input action', num2str(size(action,2)), ' instead of 2']);
            end
            
            outputState(1) =  0.1;
            outputState(2) =  (state(1)+state(2))/2;
            outputState(3) =  cos(state(1)*pi);
            outputState(4) =  cos(action(1)*pi);
            outputState(5) =  (action(1)+action(2))/2;
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
        

        
        
        
    end %end methods
    
    
end

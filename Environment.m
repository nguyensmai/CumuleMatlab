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
            
            outputState(1) =  cos(action(1)*5*pi);
            outputState(2) =  (state(1)+state(2))/2;
            outputState(3) =  cos(state(1)*pi);
            outputState(4) =  cos(action(1)*2*pi)* cos(action(2)*2*pi);
            outputState(5) =  (mean(exp(10*(state-1))) +mean(exp(10*(-state-1))))/2;
            outputState(6) =  ((action(1)+action(2))/2)^2;
            outputState(7) =  0.5;
            outputState(8) =  exp(5*(state(4)-1));
            outputState(9) =  2*exp(-(action(1)-0.5).^2/(2*10^-1))-1;  
            outputState(10) = exp(-(action(1)-0.7).^2/(2*10^-3))- exp(-(action(2)-0.4).^2/(2*10^-2));
            outputState(11) =  cos(action(1)*5*pi);
            outputState(12) =  (state(11)+state(12))/2;
            outputState(13) =  cos(state(11)*pi);
            outputState(14) =  cos(action(1)*2*pi)* cos(action(2)*2*pi);
            outputState(15) =  (mean(exp(15*(state-1))) +mean(exp(15*(-state-1))))/2;;
            outputState(16) =  ((action(1)+action(2))/2)^2;
            outputState(17) =  0.5;
            outputState(18) =  exp(7*(state(14)-1));
            outputState(19) =  2*exp(-(action(1)-0.5).^2/(2*10^-2))-1;
            outputState(20) =  exp(-(action(1)-0.7).^2/(2*10^-3))- exp(-(action(2)-0.1).^2/(2*10^-4)); 
            outputState(21) =  cos(action(1)*5*pi);
            outputState(22) =  (state(21)+state(22))/2;
            outputState(23) =  cos(state(21)*pi);
            outputState(24) =  cos(action(1)*2*pi)* cos(action(2)*2*pi);
            outputState(25) =  (mean(exp(5*(state-1))) +mean(exp(5*(-state-1))))/2;
            outputState(26) =  ((action(1)+action(2))/2)^2;
            outputState(27) =  1;
            outputState(28) =  exp(2*(state(24)-1));
            outputState(29) =  2*exp(-(action(1)-0.5).^2/(2*10^-2))-1;
            outputState(30) =  exp(-(action(1)-0.7).^2/(2*10^-3))- exp(-(action(2)-0.1).^2/(2*10^-4));
            outputState(31) =  cos(action(1)*5*pi);
            outputState(32) =  (state(31)+state(32))/2;
            outputState(33) =  cos(state(1)*pi);
            outputState(34) =  cos(action(1)*2*pi)* cos(action(2)*2*pi);
            outputState(35) =  (mean(exp(20*(state-1))) +mean(exp(20*(-state-1))))/2;
            outputState(36) =  ((action(1)+action(2))/2)^2;
            outputState(37) =  0.1;
            outputState(38) =  exp(10*(state(34)-1));
            outputState(39) =  2*exp(-(action(1)-0.5).^2/(2*10^-2))-1;
            outputState(40) = exp(-(action(1)-0.7).^2/(2*10^-3))- exp(-(action(2)-0.1).^2/(2*10^-4));
            outputState(41) =  cos(action(1)*5*pi);
            outputState(42) =  (state(1)+state(2))/2;
            outputState(43) =  cos(state(1)*pi);
            outputState(44) =  cos(action(1)*2*pi)* cos(action(2)*2*pi);
            outputState(45) =  (mean(exp(2*(state-1))) +mean(exp(2*(-state-1))))/2;;
            outputState(46) =  ((action(1)+action(2))/2)^2;
            outputState(47) =  0.5;
            outputState(48) =  exp(10*(state(44)-1));
            outputState(49) =  2*exp(-(action(1)-0.5).^2/(2*10^-2))-1;
            outputState(50) = exp(-(action(1)-0.7).^2/(2*10^-3))- exp(-(action(2)-0.1).^2/(2*10^-4));

            
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

function outputState = environment4(state, action)

if size(state,2) ~=5 
    error(['ENVIRONMENT: error in input state ', num2str(size(state,2)), ' instead of 5']);
end

if size(action,2) ~=2 
    error(['ENVIRONMENT: error in input action', num2str(size(action,2)), ' instead of 2']);    
end


outputState(1) = min(1, max(state(1)+ 0.2*action(1),-1));
outputState(2) = min(1, max(state(2)+ 0.2*action(2),-1));
outputState(3) = min(1, max(cos((state(1)+action(1))^2+(state(2)+action(2))^2),-1));
outputState(4) = min(1, max(cos(state(1)+ 0.2*action(1))^2,-1));

if state(1)+action(1)<0
    outputState(5) = 0;
elseif state(1)+action(1)<0.8
    outputState(5) = state(1)+action(1);
else
    outputState(5) = state(2)+action(2);
end

end
function outputState = environment3(state, action)

if size(state,2) ~=2 
    error('ENVIRONMENT: error in input state');
end

if size(action,2) ~=2 
    error('ENVIRONMENT: error in input action');    
end


outputState(1)= min(1, max(state(1)+action(1),-1));
outputState(2)= min(1, max(state(2)/2 + state(1)/2*cos(action(2)*pi),-1));

end
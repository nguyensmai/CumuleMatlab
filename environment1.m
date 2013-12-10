function outputState = environment1(state, action)
if state==1 && action>0
    outputState = action-2;
elseif state==-1 && action<0
    outputState = 2+action;
else outputState = state+action;

end
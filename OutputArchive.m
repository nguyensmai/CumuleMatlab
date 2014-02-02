classdef OutputArchive
    properties
        archiveMatrix
        history
        ARCHIVE_THRES  %max error threshold to be added to the archive
    end
    
    methods
        function outputArchive = OutputArchive()
            outputArchive.archiveMatrix = [];
            outputArchive.ARCHIVE_THRES = 10;  %10 means we in practive do not use thres
        end
        
        %find if the output is has already a predictor in the archive
        %returns the index in outArchive, time stored and index in pred
        function [already,outAlready, tAlready,delayAlready, iPredAlready] = findOutput(outArchive, out)
            if isempty(outArchive.archiveMatrix)
                already = [];
                outAlready = [];
                delayAlready=[];
                tAlready = [];
                iPredAlready =[];
            else
                already      = find(outArchive.archiveMatrix(:,1)==out);
                outAlready   = outArchive.archiveMatrix(already,2);
                delayAlready = outArchive.archiveMatrix(already,3);
                tAlready     = outArchive.archiveMatrix(already,4);
                iPredAlready = outArchive.archiveMatrix(already,end);
            end
        end %end function findOutput
        
        function outArchive = addElement(outArchive, pred, iPred, time)
            outArchive.archiveMatrix = [outArchive.archiveMatrix; pred(iPred).indOutDelay pred(iPred).maskOut pred(iPred).delay time iPred];
            if numel(pred(iPred).sizeHid)==2
                outArchive.history = [outArchive.history; pred(iPred).indOutDelay pred(iPred).maskOut pred(iPred).delay time iPred pred(iPred).sizeHid];
            elseif numel(pred(iPred).sizeHid)==1
                outArchive.history = [outArchive.history; pred(iPred).indOutDelay pred(iPred).maskOut pred(iPred).delay time iPred pred(iPred).sizeHid 0];
            end
        end %end function addElement
        
        function [outArchive,pred] = changeElement(outArchive,already,iPredAlready, pred,iPred,time)
            outArchive.archiveMatrix(already,:) = [ pred(iPred).indOutDelay pred(iPred).maskOut pred(iPred).delay time iPred];
            pred(iPredAlready).idFixed = -1;
            pred(iPred).idFixed = time;
            if numel(pred(iPred).sizeHid)==2
                outArchive.history = [outArchive.history; pred(iPred).indOutDelay pred(iPred).maskOut pred(iPred).delay time iPred pred(iPred).sizeHid];
            elseif numel(pred(iPred).sizeHid)==1
                outArchive.history = [outArchive.history; pred(iPred).indOutDelay pred(iPred).maskOut pred(iPred).delay time iPred pred(iPred).sizeHid 0];
            end
        end % end function changeElement
        
        % archive if good predictors
        function [outArchive,pred,already,iPredAlready] = checkErrorAndAdd(outArchive,pred,iPred,time)
            already=[];
            iPredAlready = [];
%             if numel(pred(iPred).sseRec)>2000
                meanSse1 = pred(iPred).meanError;
                out = pred(iPred).indOutDelay;
                if numel(out)==1
                    [already,~, ~,~, iPredAlready] = findOutput(outArchive, out);
                    if numel(already)==1
                        %if a predictor already predicts the same output, keep
                        %the best
                        %                          iPredAlready
                        %                          pred(iPredAlready)
                        meanSseAlready = pred(iPredAlready).meanError;
                        if meanSseAlready>meanSse1
                            [outArchive,pred] = changeElement(outArchive, already,iPredAlready, pred, iPred, time);
                        end
                    elseif isempty(already) && (meanSse1<outArchive.ARCHIVE_THRES)
                        %if a predictor is good and predicts a new output, add
                        %to archive
                        pred(iPred).idFixed = time;
                        outArchive = addElement(outArchive, pred, iPred, time);
                    elseif ~isempty(already)
                        disp('DEPRECATEBADPREDICTORSBATCH: error outArchive contains doublons');
                    end
                end
%             end
        end
        
        
        function plotArchiveError(obj,pred) %script
            figure
            nArchived = size(obj.archiveMatrix,1);
            nPlot = ceil(sqrt(nArchived));
            archiveM= sortrows(obj.archiveMatrix,1);
            for i=1:nArchived
                subplot(nPlot, nPlot,i)
                semilogy(smooth(pred(archiveM(i,end)).sseRec,10^3))
                if numel(pred(archiveM(i,end)).sizeHid) == 1
                    title(['output ', num2str(pred(archiveM(i,end)).maskOut),': 1 hidden layer ',num2str(pred(archiveM(i,end)).sizeHid)]);
                elseif numel(pred(archiveM(i,end)).sizeHid) == 2
                    title(['output ', num2str(pred(archiveM(i,end)).maskOut),': 2 hidden layers ',num2str(pred(archiveM(i,end)).sizeHid)]);
                end
            end
        end
        
        
        function output_error = plotArchiveTest(obj,env, pred,BATCH_SIZE) %script
            sMemoryTest =[];
            mt   = env.randomAction;
            st   = 2*rand(1,env.dimO)-1;
            stp1  = executeAction(env, st, mt);
            st =stp1;
            for t=1:4*BATCH_SIZE
                %     14:	Execute a motor command m chosen randomly
                mt   = env.randomAction;
                smt = [st  mt 1];
                sMemoryTest = [sMemoryTest; smt];
                
                %     15:	s(t + 1) ? read from sensorimotor data the new state.
                stp1  = executeAction(env, st, mt);
                
                %     16:	sm(t+1) ? read sensorimotor data
                st  = stp1;
            end
            
            % predictions of the archive
            figure
            archiveM= sortrows(obj.archiveMatrix,1);
            nArchived = size(obj.archiveMatrix,1);
            nPlot = ceil(sqrt(nArchived));
            for i= 1:nArchived
                subplot(nPlot, nPlot,i)
                iPred = archiveM(i,end);
                inp            = sMemoryTest(end-20-pred(iPred).delay:end-pred(iPred).delay, [pred(iPred).maskInp end]);
                target        = sMemoryTest(end-20:end, [pred(iPred).maskOut]);
                
                output_error(i) = errorInPrediction(pred(iPred),inp, target, 1);
                if numel(pred(archiveM(i,end)).sizeHid) == 1
                    title(['output ', num2str(pred(archiveM(i,end)).maskOut),': 1 hidden layer ',num2str(pred(archiveM(i,end)).sizeHid)]);
                elseif numel(pred(archiveM(i,end)).sizeHid) == 2
                    title(['output ', num2str(pred(archiveM(i,end)).maskOut),': 2 hidden layers ',num2str(pred(archiveM(i,end)).sizeHid)]);
                end
                xlim([0 20])
                ylim([-1 1])
            end
        end
        
    end %end methods
    
end
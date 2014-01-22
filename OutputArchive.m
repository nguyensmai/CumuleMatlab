classdef OutputArchive
    properties
        archiveMatrix
        ARCHIVE_THRES  %max error threshold to be added to the archive
    end
    
    methods
        function outputArchive = OutputArchive()
            outputArchive.archiveMatrix = [];
            outputArchive.ARCHIVE_THRES = 0.05;
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
        end %end function addElement
        
        function [outArchive,pred] = changeElement(outArchive,already,iPredAlready, pred,iPred,time)
            outArchive.archiveMatrix(already,:) = [ pred(iPred).indOutDelay pred(iPred).maskOut pred(iPred).delay time iPred];
            pred(iPredAlready).idFixed = -1;
            pred(iPred).idFixed = time;
        end % end function changeElement
        
         % archive if good predictors
         function [outArchive,pred,already] = checkErrorAndAdd(outArchive,pred,iPred,time)
             already=[];
             if numel(pred(iPred).sseRec)>61
                 meanSse1 = mean(pred(iPred).sseRec(end-60:end));
                 out = pred(iPred).indOutDelay;
                 if numel(out)==1
                     [already,~, ~,~, iPredAlready] = findOutput(outArchive, out);
                     if numel(already)==1
                     %if a predictor already predicts the same output, keep
                     %the best
%                          iPredAlready
%                          pred(iPredAlready)
                         meanSseAlready = mean(pred(iPredAlready).sseRec(end-60:end));
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
             end
         end
    
        
    end %end methods
    
end
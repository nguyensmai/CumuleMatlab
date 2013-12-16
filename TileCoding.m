classdef TileCoding
    
    properties
        dimension
        nbTiles
        xMin
        xMax
        nbLayers
    end %end properties
    
    
    methods
        function obj = TileCoding(dimension, nbTiles, xMin, xMax, layers)
            obj.dimension = dimension;
            obj.nbTiles = nbTiles;
            obj.xMin = xMin;
            obj.xMax = xMax;
            obj.nbLayers = layers; %now only 1 layer is supported, we only do discretisation
        end
        
        function sd = c2d(obj, sc)
            sc = max(obj.xMin, min(obj.xMax, sc));
            sd = zeros(obj.nbLayers,1);
            dx_perLayer = (obj.xMax-obj.xMin)./(obj.nbLayers*(obj.nbTiles-1));
            for iLayer = 1: obj.nbLayers
                sdVec  = ( (1.0*sc-(obj.xMin+(iLayer-1)*dx_perLayer)).* (obj.nbTiles-1) )./( obj.xMax-obj.xMin ) ;
                sdVec  = max(1,1+floor(sdVec));
                sdVec = min(obj.nbTiles, sdVec );
                sd(iLayer) = sdVec(1);
                for iDim=2:obj.dimension
                    sd(iLayer) = sdVec(iDim)+obj.nbTiles(iDim-1)*(sd(iLayer)-1);
                end
            end
        end
        
    end%end methods
    
end
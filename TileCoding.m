classdef TileCoding
    
    properties
        dimension
        nbTiles
        min
        max
        nbLayers
    end %end properties
    
    
    methods
        function obj = TileCoding(dimension, nbTiles, min, max, layers)
            obj.dimension = dimension;
            obj.nbTiles = nbTiles;
            obj.min = min;
            obj.max = max;
            obj.nbLayers = layers; %now only 1 layer is supported, we only do discretisation
        end
        
        function sd = c2d(obj, sc)
            sd = zeros(obj.nbLayers,1);
            dx_perLayer = 1./(obj.nbLayers*(obj.max-obj.min));
            for iLayer = 1: obj.nbLayers
                sd_iLayer = ( (1.0*sc-(obj.min+(iLayer-1)*dx_perLayer)).* obj.nbTiles )./( obj.max-obj.min ) ;
                sd_iLayer = floor(sd_iLayer);
                sd(iLayer) = sd_iLayer(1);
                for iDim=2:obj.dimension
                    sd(iLayer) = sd_iLayer(iDim)+obj.nbTiles(iDim-1)*sd(iLayer);
                end
            end
        end
        
    end%end methods
    
end
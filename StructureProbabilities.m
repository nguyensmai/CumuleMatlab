classdef StructureProbabilities < handle
    properties
	    distributions
	end

	methods
	    function obj = StructureProbabilities(num_problems, num_hidden_layers, default_means, default_sds)
		  obj.distributions = [];
		  for prob = 1:num_problems
			hidden_layer_probs = [];
			for hl = 1:num_hidden_layers
			  hidden_layer_probs = [hidden_layer_probs default_means(hl) default_sds(hl)];
			end
			obj.distributions = [obj.distributions; hidden_layer_probs];
		  end
		end

		function [mu, sigma] = getMuSigma(obj, problem, hidden_layer)
		  offset = 1+((hidden_layer-1)*2);
		  mu = obj.distributions(problem, offset);
		  sigma = obj.distributions(problem, offset+1);
		end

		function setMuSigma(obj, problem, hidden_layer, mu, sigma)
		  offset = 1+((hidden_layer-1)*2);
		  obj.distributions(problem, [offset offset+1]) = [mu sigma];
		end

		function updateProbabilities(obj, structure, problem, isWinner)
		  for hl = 1:size(structure, 2)
			[mu, sigma] = getMuSigma(obj, problem, hl);
			[new_mu, new_sigma] = updateProbability(obj, structure(hl), mu, sigma, isWinner);
			setMuSigma(obj, problem, hl, new_mu, new_sigma);
		  end
		end

		function [new_mu, new_sigma] = updateProbability(obj, num, mu, sigma, isWinner)
		  diff = num - mu;
		  signdiff = sign(diff);
		  if signdiff == 0
			signdiff = 1;
		  end
		  diff_sd = max(abs(diff), sigma*0.1);
		  if isWinner
			new_mu = mu + (signdiff*diff_sd);
			new_sigma = sigma/(1 + 2.0/((100*diff_sd)+1));
		  else
			new_mu = mu - (signdiff*diff_sd);
			new_sigma = sigma*(1 + 1.0/((100*diff_sd)+1));
		  end
		  if new_mu < 0
			new_mu = 0;
		  end
		end

		function sample = getSample(obj, hidden_layer)
		  [mu, sigma] = getMuSigma(obj, hidden_layer);
		  sample = floor((sigma*randn())+mu);
		end
	end
end

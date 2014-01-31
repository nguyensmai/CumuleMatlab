function test_StructureProbabilities()
  test_initialiaseStructureProbabilities();
  test_getMuSigma();
  test_setMuSigma();
  test_updateProbability();
  test_updateProbabilities();
  test_getSample();
end

function test_initialiaseStructureProbabilities()
  sp = StructureProbabilities(5, 2, [5 0], [4 1]);
  assert(isequal(size(sp.distributions), [5 4]));
  for problem = 1:5
	assert(isequal(sp.distributions(problem, :), [5 4 0 1]));
  end
end

function test_getMuSigma()
  sp = StructureProbabilities(5, 2, [5 0], [4 1]);
  [mu sigma] = getMuSigma(sp, 2, 1);
  assert(isequal([mu sigma], [5 4]));
  [mu sigma] = getMuSigma(sp, 2, 2);
  assert(isequal([mu sigma], [0 1]));
  sp.distributions(2, [1 2]) = [6 3];
  [mu sigma] = getMuSigma(sp, 2, 1);
  assert(isequal([mu sigma], [6 3]));
  [mu sigma] = getMuSigma(sp, 2, 2);
  assert(isequal([mu sigma], [0 1]));
  [mu sigma] = getMuSigma(sp, 3, 1);
  assert(isequal([mu sigma], [5 4]));
end

function test_setMuSigma()
  sp = StructureProbabilities(5, 2, [5 0], [4 1]);
  [mu sigma] = getMuSigma(sp, 2, 1);
  assert(isequal([mu sigma], [5 4]));
  [mu sigma] = getMuSigma(sp, 2, 2);
  assert(isequal([mu sigma], [0 1]));
  setMuSigma(sp, 2, 1, 4, 3);
  [mu sigma] = getMuSigma(sp, 2, 1);
  assert(isequal([mu sigma], [4 3]));
  [mu sigma] = getMuSigma(sp, 2, 2);
  assert(isequal([mu sigma], [0 1]));
  [mu sigma] = getMuSigma(sp, 3, 1);
  assert(isequal([mu sigma], [5 4]));
  setMuSigma(sp, 2, 2, 1, 1);
  [mu sigma] = getMuSigma(sp, 2, 2);
  assert(isequal([mu sigma], [1 1]));
end

function test_updateProbability()
  sp = StructureProbabilities(5, 2, [5 0], [4 1]);  
  [new_mu, new_sigma] = updateProbability(sp, 6, 5, 4, 0);
  assert(new_mu == 4.0);
  assert(round(new_sigma*1e4)/1.0e4 == 4.0396);

  [new_mu, new_sigma] = updateProbability(sp, 6, 5, 4, 1);
  assert(new_mu == 6.0);
  assert(round(new_sigma*1e4)/1.0e4 == 3.9223);
end

function test_updateProbabilities()
  sp = StructureProbabilities(5, 2, [5 0], [4 1]);
  updateProbabilities(sp, [6 2], 2, 0);
  new_probs = sp.distributions(2, :);
  for i = 1:size(new_probs, 2)
	new_probs(i) = round(new_probs(i)*1e4)/1.0e4;
  end
  assert(isequal(new_probs, [4.0 4.0396 0 1.0050]));

  updateProbabilities(sp, [6 2], 2, 0);
  new_probs = sp.distributions(2, :);
  for i = 1:size(new_probs, 2)
	new_probs(i) = round(new_probs(i)*1e4)/1.0e4;
  end
  assert(isequal(new_probs, [2.0 4.0597 0 1.0100]));
  
  updateProbabilities(sp, [6 2], 2, 1)
  new_probs = sp.distributions(2, :);
  for i = 1:size(new_probs, 2)
	new_probs(i) = round(new_probs(i)*1e4)/1.0e4;
  end
  assert(isequal(new_probs, [6.0 4.0396 2 1.0]));
end

function test_getSample()
  sp = StructureProbabilities(5, 2, [5 0], [4 1]);
  sample = getSample(sp, 2, 1);
  assert(sample > 0);

  sample = getSample(sp, 2, 2);
  assert(sample >= 0);
end
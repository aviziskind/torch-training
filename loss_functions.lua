require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

--dofile 'mycri.lua'
--dofile 'mycri_kai.lua'


getLossFunction = function(loss_function_name, arg1)

  
  local criterion

  if loss_function_name == 'margin' then
     -- This loss takes a vector of classes, and the index of the grountruth class as arguments.
     -- It is an SVM-like loss with a default margin of 1.
     criterion = nn.MultiMarginCriterion()

  elseif loss_function_name == 'nll' then
     -- This loss requires the outputs of the trainable model to be properly normalized log-probabilities, 
     -- which can be achieved using a softmax function
     --model:add(nn.LogSoftMax())
     
     -- The loss works like the MultiMarginCriterion: it takes a vector of classes,
     -- and the index of the grountruth class as arguments.
     criterion = nn.ClassNLLCriterion()

  elseif loss_function_name == 'mse' then

     -- for MSE, we add a tanh, to restrict the model's output
     --model:add(nn.Tanh())

     -- The mean-square error is not recommended for classification
     -- tasks, as it typically tries to do too much, by exactly modeling
     -- the 1-of-N distribution. For the sake of showing more examples,
     -- we still provide it here:
     criterion = nn.MSECriterion()
     criterion.sizeAverage = false
   

    elseif loss_function_name == 'wmse' then

     -- for MSE, we add a tanh, to restrict the model's output
     --model:add(nn.Tanh())

     -- The mean-square error is not recommended for classification
     -- tasks, as it typically tries to do too much, by exactly modeling
     -- the 1-of-N distribution. For the sake of showing more examples,
     -- we still provide it here:
     criterion = nn.WeightedMSECriterion(arg1)
     criterion.sizeAverage = false

     
  elseif loss_function_name == 'mycri' then
      --model:add(nn.Tanh())
      
      criterion = nn.mycri()
      
  elseif loss_function_name == 'mycri_kai' then

      --model:add(nn.Tanh())
      criterion = nn.mycri_kai()    
    
  else

     error('unknown loss function')

  end

  return criterion

end

local nn = require 'nn';
local torch = require 'torch';

torchutil = {}

local taskType = function(model) 
  local outputLayer = model.modules[#model.modules]
  if torch.isTypeOf(outputLayer, 'nn.LogSoftMax') then
    return 'classification'
  elseif torch.isTypeOf(outputLayer, 'nn.Linear') then
    return 'regression'
  else 
    local msg = "Don't know how to determine type of model with this output layer "
    msg = msg .. tostring(outputLayer)
    error(msg)
  end
end

function torchutil.predict(model, data, recorders, opt)
  local opt = opt or {}
  local gcFreq = opt.gcFreq or 100
  local task = opt.task or taskType(model)
  local verbose = opt.verbose or false

  if recorders == nil then
    recorders = {}
  end

  -- Make the model deterministic by disabling things like dropout.
  model:evaluate()

  local pred = torch.zeros(data:size(1)):float()

  for i=1,data:size(1) do
    if gcFreq > 0 and i % gcFreq == 0 then
      if verbose then
        print('garbage collection ' .. i .. '/' .. data:size(1))
      end
      -- The torchutil recorder classes call torch.cat every time they're
      -- called, which clutters up the heap.
      collectgarbage()
    end

    local output = model:forward(data[i]):float()
    if task == 'classification' then
      local val, idx = torch.max(output, 1)
      pred[i] = idx
    elseif task == 'regression' then
      pred[i] = output
    else 
      local msg = "Don't know how to determine type of model "
      msg = msg .. tostring(model)
      error(msg)
    end

    for _,recorder in pairs(recorders) do
      recorder(model, data[i])
    end
  end

  return pred
end

include('ImmutableModule.lua')
include('LookupTableInputDeleter.lua')
include('LookupTableInputReplacer.lua')
include('LookupTableInputZeroer.lua')
include('Renormers.lua')
include('Renormer.lua')
include('TemporalConvolutionRenormer.lua')
include('LookupTableRenormer.lua')
include('IndexRecorder.lua')
include('OutputRecorder.lua')
include('InputPrinter.lua')
include('SentenceCompacter.lua')
include('NumberUnique.lua')
include('WordCount.lua')

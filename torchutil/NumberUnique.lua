local NumberUnique, parent = torch.class('torchutil.NumberUnique', 'nn.Module')

-- A constructor
function NumberUnique:__init()
  parent.__init(self)
end

-- 
-- @param input: a 1D tensor from which the unique
-- elements are extracted to a table and counted.
-- Returning a Tensor with the count.
function NumberUnique:updateOutput(input)
  tbl = {}
  count = 0
  for i =1,input:size(1) do 
    if tbl[input[i]] == nil then
      count = count +1
      tbl[i] = input[i] 
    end
  end  
  return torch.randn(1):fill(count)
end

--
-- @param input
-- @param gradOutput
function NumberUnique:updateGradInput(input, gradOutput)
-- Copied from nn/Sum.lua
-- zero-strides dont work with MKL/BLAS, so
-- dont set self.gradInput to zero-stride tensor.
-- Instead, do a deepcopy
  local size = input:size()
  size[self.dimension] = 1
  gradOutput = gradOutput:view(size)
  self.gradInput:resizeAs(input)
  self.gradInput:copy(gradOutput:expandAs(input))
  return self.gradInput
end

-- 
-- @param input
-- @param gradOutput
function NumberUnique:accGradParameters(input, gradOutput)
  parent.accGradParameters(input, gradOutput)
end

--
function NumberUnique:reset()
end

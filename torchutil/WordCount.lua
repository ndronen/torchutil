local WordCount, parent = torch.class('torchutil.WordCount', 'nn.Module')

-- A constructor
function WordCount:__init(dimension)
  parent.__init(self)
  dimension = dimension or 1
  self.dimension = dimension
end

-- 
-- @param input: a 1D tensor from which the count 
-- of the elements is extracted.
-- Returning a Tensor with the word count.
function WordCount:updateOutput(input)
  wc = input:nElement()
  --making the output a tensor and not a number
  wc = torch.randn(1):fill(wc) 
  return wc
end

--
-- @param input
-- @param gradOutput
function WordCount:updateGradInput(input, gradOutput)
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
function WordCount:accGradParameters(input, gradOutput)
  parent.accGradParameters(input, gradOutput)
end

--
function WordCount:reset()
end

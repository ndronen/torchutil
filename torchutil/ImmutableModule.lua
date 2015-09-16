local ImmutableModule, parent = torch.class('torchutil.ImmutableModule', 'nn.Module')

-- A constructor
function ImmutableModule:__init(delegate)
  parent.__init(self)
  self.delegate = delegate
end

-- 
-- @param input 
function ImmutableModule:updateOutput(input)
  self.output = self.delegate:updateOutput(input)
  return self.output
end

--
-- @param input
-- @param gradOutput
function ImmutableModule:updateGradInput(input, gradOutput)
  self.gradInput = self.delegate:updateGradInput(input, gradeOutput)
  return self.gradInput
end

-- 
-- @param input
-- @param gradOutput
function ImmutableModule:accGradParameters(input, gradOutput)
end

--
function ImmutableModule:reset()
  self.delegate:reset()
end

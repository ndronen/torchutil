local Skeleton, parent = torch.class('torchutil.Skeleton', 'nn.Module')

-- A constructor
function Skeleton:__init()
  parent.__init(self)
end

-- 
-- @param input 
function Skeleton:updateOutput(input)
end

--
-- @param input
-- @param gradOutput
function Skeleton:updateGradInput(input, gradOutput)
end

-- 
-- @param input
-- @param gradOutput
function Skeleton:accGradParameters(input, gradOutput)
  parent.accGradParameters(input, gradOutput)
end

--
function NewClass:reset()
end

local LookupTableInputReplacer, parent = torch.class('torchutil.LookupTableInputReplacer', 'nn.Module')

--[[
Selects indices of the input with some fixed probability p and replaces
them with random indices.

If an instance of this module is used in a Container, it should be
followed by a LookupTable layer.   That restriction does not apply if
it is used as a standalone object.
--]]
-- @param p The probability of replacing an element of the input.
-- @param index The index in the lookup table of the zero vector.
function LookupTableInputReplacer:__init(p, maxIndex)
  parent.__init(self)
  if (p < 0) or (p > 1) then
    error(string.format('p must be in the range 0-1: %f', p))
  end
  self.p = p
  self.index = index
end

-- 
-- @param input 
function LookupTableInputReplacer:updateOutput(input)
  self.output = input:clone()
  if self.train then
    local mask = input:clone():bernoulli(1-self.p)
    local newIndices = torch.rand(mask:eq(0):sum()):mul(10):add(1):int()
    self.output[mask:eq(0)] = newIndices
  end
  return self.output
end

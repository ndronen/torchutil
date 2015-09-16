local LookupTableInputZeroer, parent = torch.class('torchutil.LookupTableInputZeroer', 'nn.Module')

--[[
Replaces lookup table indices with the index of a (presumed) zero
vector with some fixed probability p.  This is analagous to dropping
inputs randomly, but -- unlike LookuptableInputDeleter -- preserves the
distances among inputs.

This module should be the first layer of a network and should be followed
by a LookupTable layer.  
--]]
-- @param p The probability of dropping element of the input.
-- @param index The index in the lookup table of the zero vector.
function LookupTableInputZeroer:__init(p, index)
  parent.__init(self)
  if (p < 0) or (p > 1) then
    error(string.format('p must be in the range 0-1: %f', p))
  end
  self.p = p
  self.index = index
end

-- 
-- @param input 
function LookupTableInputZeroer:updateOutput(input)
  self.output = input:clone()
  if self.train then
    local mask = input:clone():bernoulli(1-self.p)
    self.output[mask:eq(0)] = self.index
  end
  return self.output
end

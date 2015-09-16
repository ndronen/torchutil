local LookupTableInputDeleter, parent = torch.class('torchutil.LookupTableInputDeleter', 'nn.Module')

--[[
Deletes lookup table indices with some probability p during training.
As long as the input is non-empty, the output is guaranteed to be
non-empty.  If all words are selected to be deleted, either the input
is passed through unchanged or another attempt is made to transform it.

This module should be the first layer of a network and should be followed
by a LookupTable layer.  
--]]
-- @param index The probability with which to delete an element of an input.
-- @param maxTries The number of times to attempt to randomly delete words from the input before passing the input through.
function LookupTableInputDeleter:__init(p, maxTrys)
  parent.__init(self)
  self.p = p
  self.maxTrys = maxTrys
  if (p < 0) or (p > 1) then
    error(string.format('p must be in the range 0-1: %f', p))
  end
  if maxTrys < 1 then
    error(string.format('maxTrys must be > 0: %d', maxTrys))
  end
end

-- 
-- @param input 
function LookupTableInputDeleter:updateOutput(input)
  if input:nDimension() > 1 then
    error('inputs to LookupTableInputDeleter must be 1-d, not %d-d',
      input:nDimension())
  end
  self.output = input
  if self.train then
    for i=1,self.maxTrys do
      local mask = input:clone():bernoulli(self.p)
      local output = input[mask:byte()]
      if output:nElement() > 0 then
        self.output = output
        break
      end
    end
  end
  return self.output
end

local SentenceCompacter, parent = torch.class('torchutil.SentenceCompacter', 'nn.Module')

require 'torchx';

--[[
Takes a potentially zero-padded 1-d tensor as input and eliminates any
unnecessary padding.  This speeds up the training time for temporal
convolutional networks.

This module should be followed by a LookupTable layer.  
--]]
-- @param padding The number of leading and trailing padding elements.
-- @param zeroVectorIndex The index for padding and the unknown word.
function SentenceCompacter:__init(padding, zeroVectorIndex)
  parent.__init(self)
  self.padding = padding
  self.zeroVectorIndex = zeroVectorIndex
end

-- 
-- @param input 
function SentenceCompacter:updateOutput(input)
  if input:nDimension() > 1 then
    error('inputs to SentenceCompacter must be 1-d, not %d-d',
      input:nDimension())
  end
  local mask = input:ne(self.zeroVectorIndex)
  -- Find the indices of the words that are not padding or unknown words.
  local indices = torch.find(mask, 1)
  if #indices >= 2 then
    local firstWord = indices[1]
    local lastWord = indices[#indices]
    self.output = input[{{firstWord-self.padding, lastWord+self.padding}}]
  else
    self.output = input
  end
  return self.output
end

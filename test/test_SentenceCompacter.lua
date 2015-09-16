require 'nn'
require 'torchutil'

local tester = torch.Tester()
local compactertest = {}

local zeroVectorIndex = 10
local sentLen = 13

function compactertest.FullSentenceUnchanged()
  for padding=0,3 do
    x = (torch.rand(sentLen) * (zeroVectorIndex-1)):int() + 1
    for i=1,padding do
      x[i] = zeroVectorIndex
      x[sentLen+1-i] = zeroVectorIndex
    end
    compacter = torchutil.SentenceCompacter(padding, zeroVectorIndex)
    xBefore = x:clone()
    xActual = compacter:forward(x)
    tester:assertTensorEq(
      xBefore,
      xActual,
      0,
      'sentence changed unnecessarily')
  end
end

function compactertest.ShortSentenceChanged()
  for padding=0,3 do
    x = (torch.rand(sentLen) * (zeroVectorIndex-1)):int() + 1
    local j = 0
    for i=1,padding do
      x[i] = zeroVectorIndex
      x[sentLen+1-i] = zeroVectorIndex
      j = i
    end
    -- Put a zero entry right before the padding.
    x[sentLen-j] = zeroVectorIndex
    -- Put a zero entry somewhere in the middle of the sentence.
    x[sentLen-j-2] = zeroVectorIndex
    compacter = torchutil.SentenceCompacter(padding, zeroVectorIndex)
    xBefore = x:clone()
    xExpected = torch.IntTensor(sentLen - 1)
    k = 0
    for i=1,x:size(1) do
      if i ~= sentLen-j then
        k = k + 1
        xExpected[k] = x[i]
      end
    end
    xActual = compacter:forward(x)
    tester:assertTensorEq(
      xExpected,
      xActual,
      0,
      'sentence was not compacted correctly')
  end
end

tester:add(compactertest)

function nn.testcompacter(tests)
  local oldtype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  math.randomseed(os.time())
  tester:run(tests)
  torch.setdefaulttensortype(oldtype)
end

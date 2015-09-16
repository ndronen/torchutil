require 'nn'
require 'torchutil'

local tester = torch.Tester()
local renormingtest = {}

local nInputs = 10
local nNeurons = 3

function renormingtest.RenormerNoChangeForHighMaxNorm()
  local module = nn.Linear(nInputs, nNeurons)

  local maxNorm = 10000
  local renormer = torchutil.Renormer(module, maxNorm)
  local weightBefore = module.weight:clone()
  renormer:renorm()

  tester:assertTensorEq(
    weightBefore,
    module.weight,
    1e-4,
    string.format(
      'weights changed unnecessarily when maxNorm==%d',
      maxNorm))
end

function renormingtest.RenormerNormsCorrectAfterRenorm()
  local module = nn.Linear(nInputs, nNeurons)

  local maxNorm = 0.2
  local renormer = torchutil.Renormer(module, maxNorm)

  local normsBefore = module.weight:norm(2, 2)
  renormer:renorm()
  normsAfter = module.weight:norm(2, 2)

  tester:assert(normsAfter:lt(normsBefore))
end

function renormingtest.TemporalRenormerNormsCorrectAfterRenorm()
  local wordDims = 25
  local inputFrameSize = wordDims
  local nFilters = 10
  local outputFrameSize = nFilters
  local filterWidth = 3
  local stride = 1
  local module = nn.TemporalConvolution(
      inputFrameSize, outputFrameSize, filterWidth, stride)
  -- Make the norms of the filter components large, so the renorming
  -- operation will affect all of them.
  module:reset(10)

  local maxComponentNorm = 1
  local renormer = torchutil.TemporalConvolutionRenormer(
      module, maxComponentNorm)
  renormer:renorm()

  -- After renorming, the norms should be the square root of filter width.
  local normsActual = module.weight:norm(2, 2)
  local normsExpected = normsActual:clone():fill(math.sqrt(filterWidth))

  tester:assertTensorEq(
    normsExpected,
    normsActual,
    1e-4,
    'temporal convolution layer weights not renormed correctly')
end

function renormingtest.LookupTableRenormerCPUNormsCorrectAfterRenorm()
  local nWords = 1000
  local wordDims = 25
  local module = nn.LookupTable(nWords, wordDims)
  -- Make the norms of the filter components large, so the renorming
  -- operation will affect all of them.
  module:reset(10)

  local maxNorm = 1
  local renormer = torchutil.LookupTableRenormer(module, maxNorm)
  renormer:renorm()

  local normsActual = module.weight:norm(2, 2)
  local normsExpected = normsActual:clone():fill(maxNorm)

  tester:assertTensorEq(
    normsExpected,
    normsActual,
    1e-4,
    'lookup table weights not renormed correctly')
end

tester:add(renormingtest)

function nn.testrenorming(tests)
  local oldtype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  math.randomseed(os.time())
  tester:run(tests)
  torch.setdefaulttensortype(oldtype)
end

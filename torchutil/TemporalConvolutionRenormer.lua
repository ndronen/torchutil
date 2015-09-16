local TemporalConvolutionRenormer = torch.class('torchutil.TemporalConvolutionRenormer')

-- A constructor
function TemporalConvolutionRenormer:__init(module, maxNorm, opts)
  assert (module ~= nil)
  assert (maxNorm > 0)
  self.module = module
  self.maxNorm = maxNorm

  opts = opts or {}
  self.p = opts.p or 2
  self.dim = opts.dim or 1
  if opts['scope'] ~= nil then
    self.scope = opts['scope']
  else
    self.scope = 'components'
  end

  -- Possibly needlessly strict constraints on p.
  assert ((self.p == 1) or (self.p == 2))

  -- Assume module's 'weight' is a 2d tensor.
  assert ((self.dim == 1) or (self.dim == 2))

  assert ((self.scope == 'components') or (self.scope == 'all'))
end

function TemporalConvolutionRenormer:renorm()
  if self.renormComponents then
    self:renormComponents()
  else
    self:renormAll()
  end
end

function TemporalConvolutionRenormer:renormAll()
    self.module.weight:renorm(self.p, self.dim, self.maxNorm)
end

function TemporalConvolutionRenormer:renormComponents() 
  local step = self.module.inputFrameSize
  local tmp = torch.Tensor():typeAs(self.module.weight)

  --[[
  Renorm each component of the weights.  By component we mean the slice
  of a filter that corresponds to a full word representation in the input.
  In the nn package, a filter is represented as a 1-row matrix with
  width inputFrameSize * kW.  Efficiently renorming each of the kW
  components in a filter is accomplished by setting a view onto the
  storage of each of those components and renorming the views.

  Renorming each component can be used to ensure that the word
  representations are embedded in cosine space, and that the convolutional
  operation happens in cosine space as well.
  --]]
  for i=1,self.module.outputFrameSize do
    filterOffset = (i - 1) * (step * self.module.kW)
    for j=1,self.module.kW do
      offset = filterOffset + (step * (j - 1)) + 1
      tmp:set(self.module.weight:storage(), offset, torch.LongStorage({1, step}))
      -- TODO: if this fails in CUDA/GPU mode, then the storage will need to be copied
      -- to the CPU before renorming, and back to the GPU afterward.
      tmp:renorm(self.p, self.dim, self.maxNorm)
    end
  end
end

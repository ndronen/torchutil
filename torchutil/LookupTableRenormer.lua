local LookupTableRenormer = torch.class('torchutil.LookupTableRenormer')

-- A constructor
function LookupTableRenormer:__init(module, maxNorm, opts)
  assert (module ~= nil)
  assert (maxNorm > 0)
  self.module = module
  self.maxNorm = maxNorm

  opts = opts or {}
  self.p = opts.p or 2
  self.dim = opts.dim or 1

  -- Possibly needlessly strict constraints on p.
  assert ((self.p == 1) or (self.p == 2))

  -- Assume module's 'weight' is a 2d tensor.
  assert ((self.dim == 1) or (self.dim == 2))
end

function LookupTableRenormer:renorm()
  if torch.isTypeOf(self.module, 'nn.LookupTableGPU') then
    -- Copy to the CPU and back again.  This is a workaround for what
    -- is either a bug in fbcunnn's LookupTableGPU or some peculiarity
    -- of CUDA.  Under some conditions that I've not isolated yet, the
    -- renorming will fail if you try to do it on the GPU.
    tmp = self.module.weight:clone():float()
    tmp:renorm(self.p, self.dim, self.maxNorm)
    self.module.weight = tmp:cuda()
  else
    self.module.weight:renorm(self.p, self.dim, self.maxNorm)
  end
end

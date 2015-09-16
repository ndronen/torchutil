local Renormer = torch.class('torchutil.Renormer')

-- A constructor
function Renormer:__init(module, maxNorm, opts)
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

function Renormer:renorm()
  self.module.weight:renorm(self.p, self.dim, self.maxNorm)
end

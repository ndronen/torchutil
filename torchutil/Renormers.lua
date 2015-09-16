local Renormers, parent = torch.class('torchutil.Renormers', 'nn.Container')

-- A constructor
function Renormers:__init()
  parent.__init(self)
  self.renormers = {}
end

-- 
-- @param renormer
function Renormers:add(renormer) 
  table.insert(self.renormers, renormer)
end

--
-- @param index
function Renormers:get(index)
  return self.renormers[index]
end

--
function Renormers:size() 
  return #self.renormers
end

--
function Renormers:renorm()
  for i,renormer in ipairs(self.renormers) do
    renormer:renorm()
  end
end

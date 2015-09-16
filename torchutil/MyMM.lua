--[[ Module to....
]]

local MyMM, parent = torch.class('nn.MyMM', 'nn.Module')

--[[ The constructor....
]]
function MyMM:__init(nFilters, wordDim, filterWidth)
  parent.__init(self)

  -- ?
  self.gradInput = {torch.Tensor(), torch.Tensor()}
  self.output = torch.Tensor(nFilters, filterWidth, filterWidth)

  -- Same fields as in nn.Linear.
  self.weight = torch.Tensor(nFilters, wordDim, filterWidth)
  self.bias = torch.Tensor(nFilters)

  self.gradWeight = torch.Tensor(nfilters, wordDim, filterWidth)
  self.gradBias = torch.Tensor(nFilters)
end

function MyMM:updateOutput(input)
  assert(input:nDimension() == 3)
  assert(self.weight:size(1) == input:size(1))
  assert(self.weight:size(2) == input:size(3))
  assert(self.weight:size(3) == input:size(2))
  self.output:resize(input:size(1), input:size(2), input:size(2))
  self.output:bmm(input, self.weight)
  return self.output
end

function MyMM:updateGradInput(input, gradOutput)
  assert(#input == 2, 'input must be a pair of tensors')
  local a, b = unpack(input)
  self.gradInput[1]:resizeAs(a)
  self.gradInput[2]:resizeAs(b)

  assert(gradOutput:nDimension() == 2 or gradOutput:nDimension() == 3, 'arguments must be a 2D or 3D Tensor')

  local h_dim, w_dim, f
  if gradOutput:nDimension() == 2 then
    assert(a:nDimension() == 2, 'first input tensor must be 2D')
    assert(b:nDimension() == 2, 'second input tensor must be 2D')

    h_dim, w_dim = 1, 2
    f = "mm"
  else
    assert(a:nDimension() == 3, 'first input tensor must be 3D')
    assert(b:nDimension() == 3, 'second input tensor must be 3D')

    h_dim, w_dim = 2, 3
    f = "bmm"
  end

  if self.transA == self.transB then
    a = a:transpose(h_dim, w_dim)
    b = b:transpose(h_dim, w_dim)
  end

  if self.transA then
    self.gradInput[1][f](self.gradInput[1], b, gradOutput:transpose(h_dim, w_dim))
  else
    self.gradInput[1][f](self.gradInput[1], gradOutput, b)
  end

  if self.transB then
    self.gradInput[2][f](self.gradInput[2], gradOutput:transpose(h_dim, w_dim), a)
  else
    self.gradInput[2][f](self.gradInput[2], a, gradOutput)
  end

  return self.gradInput
end

function MyMM:accGradParameters(input, gradOutput, scale)
end

-- we do not need to accumulate parameters when sharing
MyMM.sharedAccUpdateGradParameters = MyMM.accUpdateGradParameters

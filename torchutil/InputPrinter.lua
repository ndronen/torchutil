local InputPrinter = torch.class('torchutil.InputPrinter', 'nn.Module')

local torch = require('torch');

-- A constructor
function InputPrinter:__init(name)
  self.name = name
end

function InputPrinter:updateOutput(input)
  print(self.name)
  print(input:size())
  self.output = input
  return self.output
end

function InputPrinter:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end

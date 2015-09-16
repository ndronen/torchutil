local IndexRecorder = torch.class('torchutil.IndexRecorder')

local torch = require('torch');

-- A constructor
function IndexRecorder:__init(module)
  self.module = module
  self.recording = torch.Tensor()
end

function IndexRecorder:getRecording() 
  return self.recording
end

function IndexRecorder:__call__(model, data)
  if self.recording:dim() == 0 then
    self.recording:resize(self.module.indices:size())
    self.recording:copy(self.module.indices)
  else
    self.recording = self.recording:cat(self.module.indices:double(), 1)
  end
end


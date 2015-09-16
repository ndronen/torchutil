local OutputRecorder = torch.class('torchutil.OutputRecorder')

local torch = require('torch');

-- A constructor
function OutputRecorder:__init(module)
  self.module = module
  self.recording = torch.Tensor()
end

function OutputRecorder:getRecording()
  return self.recording
end

function OutputRecorder:__call__(model, data)
  if self.recording:dim() == 0 then
    self.recording:resize(self.module.output:size())
    self.recording:copy(self.module.output)
  else
    self.recording = self.recording:cat(self.module.output:double(), 1)
  end
end

package = "torchutil"
version = "scm-1"

source = {
    url = "git://github.com/ndronen/torchutil"
}

description = {
   summary = "Extensions to and utilities for Torch.",
   detailed = [[
        Miscellaneous utilities that are somewhat useful.
   ]],
   homepage = "https://github.com/ndronen/torchutil",
   license = "MIT/BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0"
}

build = {
    type = "builtin",
    modules = {
        torchutil = "torchutil/init.lua",
        ["torchutil.LookupTableInputDeleter"] = "torchutil/LookupTableInputDeleter.lua",
        ["torchutil.LookupTableInputReplacer"] = "torchutil/LookupTableInputReplacer.lua",
        ["torchutil.LookupTableInputZeroer"] = "torchutil/LookupTableInputZeroer.lua",
        ["torchutil.Renormers"] = "torchutil/Renormers.lua",
        ["torchutil.Renormer"] = "torchutil/Renormer.lua",
        ["torchutil.TemporalConvolutionRenormer"] = "torchutil/TemporalConvolutionRenormer.lua",
        ["torchutil.LookupTableRenormer"] = "torchutil/LookupTableRenormer.lua",
        ["torchutil.ImmutableModule"] = "torchutil/ImmutableModule.lua",
        ["torchutil.IndexRecorder"] = "torchutil/IndexRecorder.lua",
        ["torchutil.OutputRecorder"] = "torchutil/OutputRecorder.lua",
        ["torchutil.InputPrinter"] = "torchutil/InputPrinter.lua",
        ["torchutil.SentenceCompacter"] = "torchutil/SentenceCompacter.lua",
        ["torchutil.NumberUnique"] = "torchutil/NumberUnique.lua",
        ["torchutil.WordCount"] = "torchutil/WordCount.lua"
    }
}

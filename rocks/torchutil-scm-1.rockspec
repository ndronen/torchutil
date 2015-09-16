package = "torchutil"
version = "scm-1"

source = {
    url = "git://github.com/ndronen/torchutil",
    tag = "HEAD",
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
        kttorch = "kttorch/init.lua",
        ["kttorch.LookupTableInputDeleter"] = "kttorch/LookupTableInputDeleter.lua",
        ["kttorch.LookupTableInputReplacer"] = "kttorch/LookupTableInputReplacer.lua",
        ["kttorch.LookupTableInputZeroer"] = "kttorch/LookupTableInputZeroer.lua",
        ["kttorch.Renormers"] = "kttorch/Renormers.lua",
        ["kttorch.Renormer"] = "kttorch/Renormer.lua",
        ["kttorch.TemporalConvolutionRenormer"] = "kttorch/TemporalConvolutionRenormer.lua",
        ["kttorch.LookupTableRenormer"] = "kttorch/LookupTableRenormer.lua",
        ["kttorch.ImmutableModule"] = "kttorch/ImmutableModule.lua",
        ["kttorch.IndexRecorder"] = "kttorch/IndexRecorder.lua",
        ["kttorch.OutputRecorder"] = "kttorch/OutputRecorder.lua",
        ["kttorch.InputPrinter"] = "kttorch/InputPrinter.lua",
        ["kttorch.SentenceCompacter"] = "kttorch/SentenceCompacter.lua",
        ["kttorch.NumberUnique"] = "kttorch/NumberUnique.lua",
        ["kttorch.WordCount"] = "kttorch/WordCount.lua"
    }
}

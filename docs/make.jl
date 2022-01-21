using PyCallChainRules
using Documenter

makedocs(;
    modules=[PyCallChainRules],
    authors="rejuvyesh",
    repo="https://github.com/rejuvyesh/PyCallChainRules.jl/blob/{commit}{path}#L{line}",
    sitename="PyCallChainRules.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://rejuvyesh.github.io/PyCallChainRules.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],    
)

deploydocs(;
    repo="github.com/rejuvyesh/PyCallChainRules.jl",
)
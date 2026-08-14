using Documenter
using NestedThreading

DocMeta.setdocmeta!(
    NestedThreading, :DocTestSetup, :(using NestedThreading); recursive = true
)

makedocs(;
    modules = [NestedThreading],
    format = Documenter.HTML(),
    sitename = "NestedThreading.jl",
    repo = "https://github.com/hakkelt/NestedThreading.jl/blob/{commit}{path}#{line}",
    authors = "Tamás Hakkel",
    pages = [
        "Home" => "index.md",
        "Composition rules" => "composition.md",
        "Adding a library" => "extending.md",
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/hakkelt/NestedThreading.jl", target = "build")

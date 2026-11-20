using Documenter
using NestedThreading

DocMeta.setdocmeta!(
    NestedThreading, :DocTestSetup, :(using NestedThreading); recursive = true
)

makedocs(;
    modules = [NestedThreading],
    format = Documenter.HTML(;
        edit_link = "master", repolink = "https://github.com/hakkelt/NestedThreading.jl"
    ),
    sitename = "NestedThreading.jl",
    repo = "https://github.com/hakkelt/NestedThreading.jl/blob/{commit}{path}#{line}",
    authors = "Tamás Hakkel",
    pages = [
        "Home" => "index.md",
        "How-to guides" => [
            "guides/loops.md",
            "guides/scopes.md",
            "guides/extending.md",
        ],
        "Design" => "design.md",
        "Reference" => ["rules.md", "libraries.md", "api.md"],
    ],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/hakkelt/NestedThreading.jl", target = "build")

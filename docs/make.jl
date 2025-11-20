push!(LOAD_PATH,"../src/")

using Documenter, StaticArrays, VisionGeometry, MeshRender

makedocs(sitename = "MeshRender.jl",
			checkdocs = :exports,
         modules  = [MeshRender],
         pages = ["Home" => "index.md"])

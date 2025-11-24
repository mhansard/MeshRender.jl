# MeshRender Julia package

A configurable and extensible OpenGL / GLFW renderer, for triangle-mesh models. Each mesh is represented by a vector `F` of `SVector{3,Int}` face-indices, along with corresponding vectors of `SVector{3,<:Real}` vertices `V` and (per-vertex) normals `N`.

Texture maps, per-vertex, and per-face colours are supported. Sub-typing of the basic renderer is possible, including custom vertex and fragment shaders.

## Examples

```
# Render two meshes (format as described above)

julia> rend = Renderer([F1,F2], [V1,V2], [N1,N2])
julia> rend()

# Save a capture of the current view
julia> rend("view.png")

# Textured OBJ data from https://3d.si.edu/collections/apollo11

julia> objs = ["apollo/x3d-cm-exterior-shell-90k-uvs.obj", 
               "apollo/x3d-cm-exterior-top-160k-uvs.obj"]

julia> pngs = ["apollo/x3d-cm-exterior-shell-90k-comp-4k.png",
               "apollo/x3d-cm-exterior-top-160k-comp-4k.png"]

# Render in a 1000×1000 viewport
julia> render_objs(objs, pngs; view_size=(1000,1000))
```

## Keyboard controls
- `Tab`: Show next mesh.
- `Backspace`: Show previous mesh.
- `Shift`+`Tab`: Add next mesh.
- `Shift`+`Backspace`: Remove previous mesh.
- `c`, `t`, `d`, `p`: Render colour, texture, depth, and points (where available).
- `o`: Render colour at 50% opacity.
- `s`: Save image to `meshrender.png` in current directory.
- `Esc`: Quit.

## Installation and loading

Requires a working OpenGL4 installation

Clone the repository, and run the following in Julia:
* `using Pkg` 
* `Pkg.develop(path="/yourpath/MeshRender.jl")`
* `using MeshRender`

Build the local documentation
* `cd /yourpath/MeshRender.jl/docs`
* `julia --project make.jl`

View the local documentation at `/yourpath/MeshRender.jl/docs/build/index.html`.

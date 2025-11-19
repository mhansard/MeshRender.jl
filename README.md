# MeshRender

A configurable and extensible OpenGL / GLFW renderer, for triangle-mesh models. Each mesh is represented by a vector `F` of `SVector{3,Int}` face-indices, along with corresponding vectors of `SVector{3,<:Real}` vertices `V` and (per-vertex) normals `N`.

Texture maps, per-vertex, and per-face colours are supported. Sub-typing of the basic renderer is possible, including custom vertex and fragment shaders.

## Schematic example

```
# Render two meshes in window of size w×h
rend = Renderer((w,h), [F1,F2], [V1,V2], [N1,N2])
rend()

# Save a capture of the current view
rend("view.png")
```

## Textured OBJ file example

```
# Run the wrapper render_objs() on data from https://3d.si.edu/collections/apollo11

objs = ["apollo/x3d-cm-exterior-shell-90k-uvs.obj", 
         "apollo/x3d-cm-exterior-top-160k-uvs.obj"]

pngs = ["apollo/x3d-cm-exterior-shell-90k-comp-4k.png",
        "apollo/x3d-cm-exterior-top-160k-comp-4k.png"]

MeshRender.render_objs(objs,pngs)

```
## Keyboard controls
- `Tab`: Show next mesh
- `Backspace`: Show previous mesh
- `Shift`+`Tab`: Add next mesh
- `Shift`+`Backspace`: Remove previous mesh
- `c`, `t`, `d`, `p`: Render colour, texture, depth, points (where available)
- `o`: Render colour at 50% opacity
- `s`: Save image to `meshrender.png` in current directory.
- `Esc`: Quit

# Installation and loading

Clone the repository, and run the following in Julia:
* `Pkg.dev("/yourpath/MeshRender.jl")`
* `using MeshRender`

Build the local documentation
* `cd /yourpath/MeshRender.jl/docs`
* `julia --project make.jl`

View the local documentation at `/yourpath/MeshRender.jl/docs/build/index.html`.

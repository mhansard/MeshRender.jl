# MeshRender

A configurable and extensible OpenGL renderer, for triangle-mesh models.

## Schematic example

```
# Render two meshes
rend = Renderer((w,h), [F1,F2], [V1,V2], [N1,N2])
rend()
# Offscreen version
rend("view.png")
```

## Real example

```
# Run the wrapper render_objs() on data from https://3d.si.edu/collections/apollo11

objs = ["apollo/x3d-cm-exterior-shell-90k-uvs.obj", 
         "apollo/x3d-cm-exterior-top-160k-uvs.obj"]

pngs = ["apollo/x3d-cm-exterior-shell-90k-comp-4k.png",
        "apollo/x3d-cm-exterior-top-160k-comp-4k.png"]

MeshRender.render_objs(objs,pngs)

```

# Installation


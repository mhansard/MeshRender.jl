module MeshRender

# MeshRender.render_obj_meshes(["apollo/x3d-cm-exterior-shell-90k-uvs.obj", "apollo/x3d-cm-exterior-top-160k-uvs.obj"], ["apollo/x3d-cm-exterior-shell-90k-comp-4k.png", "apollo/x3d-cm-exterior-top-160k-comp-4k.png"])

using Distributed, StaticArrays, LinearAlgebra, StatsBase, GLFW, ModernGL, FileIO,
Images, Colors, ImageTransformations, Interpolations

import GeometryBasics

export Renderer, viewing!, render_objs

# Auxiliary files
include(pkgdir(ModernGL, "test", "util.jl"))

""" Robust computation of angle between vectors
"""
function angle(v1,v2)
	u1 = normalize(v1)
	u2 = normalize(v2)
	2.0 * atan(norm(u1-u2), norm(u1+u2))
end

""" Rotation matrix from angle and axis
"""
function rotation(t::Real, v::SVector{3,<:Real})
   u = normalize(v)
	S = u * transpose(u)
	Q = stack(cross.([u,u,u], eachcol(I(3))))
	# Rodrigues formula
   cos(t)*I + sin(t)*Q + (1.0-cos(t))*S
end

""" Construct OpenGL camera matrix from [near,far] limits (unsigned),
    field of view (degrees), and aspect ratio.
"""
function perspective(clip::Tuple{Float64,Float64}, fov_v_deg::Float64, aspect::Float64=1.0)
   # Homogeneous perspective
	fov_v_rad = fov_v_deg * π/180.0
   f = 1.0/tan(fov_v_rad/2.0)
   d = clip[2] - clip[1]
   cam = zeros(4,4)
   cam[1,1] = f / aspect
   cam[2,2] = f
   cam[3,3] = -sum(clip) / d
   cam[3,4] = -2.0*prod(clip) / d
   cam[4,3] = -1.0
   cam[4,4] =  0.0
   cam
end

""" Construct modelview matrix from coordinates of camera, 
    target, and up-vector.
"""
function modelview(cam::SVector{3,<:Number}, at::SVector{3,<:Number}; up::AbstractVector=[0.0; 1.0; 0.0],
	                rotation::Matrix{Float64}=I, scale::Float64=1.0)
   # Modelview matrix
   v = normalize(at - cam)
   n = normalize(cross(v,up))
   u = normalize(cross(n,v))
   # Homogeneous form with translation
   M = [ n' -dot(n,cam);
         u' -dot(u,cam);
        -v'  dot(v,cam); 
         0 0 0 1 ]
   Diagonal([scale,scale,scale,1]) * M * [rotation zeros(3,1); [0 0 0 1]]
end

""" Compute z-component of the generalized arcball vector.
"""
function arcball_depth(v::SVector{2,<:Number}, r1::Float64=1.0)
	r_sqr = v[1]^2 + v[2]^2
	(r_sqr <= 0.5*r1^2) ? sqrt(r1^2-r_sqr) : (0.5*r1^2)/sqrt(r_sqr)
end

""" Compute the generalized arcball vector. 
"""
function arcball_vector(image_size::SVector{2,<:Number}, cursor_pos::SVector{2,<:Number})
	# Center and radial 2D vector
	c = (image_size .- 1.0) ./ 2.0
	q = (cursor_pos .- c) ./ (min(image_size...)-1.0)
	# Radial 3D vector
	SVector{3,<:Number}(q[1], -q[2], arcball_depth(q))
end

""" Initialize GL vector from concatenated array
"""
function gl_vec(M::AbstractArray, gl_type::DataType=GLfloat)
   Array{gl_type,1}(vec(M))
end

""" Byte offset as multiple of type size
"""
function ptr_offset(n::Int, T::DataType=GLfloat)
   Ptr{Cvoid}(n * sizeof(T))
end

function gl_check(msg::String="")
	err = glGetError()
	if err != GL_NO_ERROR
		println("Failed: $(GLENUM(err).name)" * msg)
	end
end

abstract type AbstractRenderer end 

@enum Render colour=1 depth=2 points=3 texture=4

""" Low-level OpenGL representation of a single mesh.
"""
struct GLBuffers
	vao::GLuint
	vbo::GLuint
	ibo::GLuint
	tex::GLuint
	n::GLuint
	function GLBuffers(n::Int)
		new(glGenVertexArray(), glGenBuffer(), glGenBuffer(), glGenTexture(), n)
	end
end

""" Complete OpenGL representation, containing vertex, index, framebuffer and texture data.
"""
mutable struct GLData

	program::GLuint
	mesh_buffers::Vector{GLBuffers}
	point_buffers::Vector{GLBuffers}
	image_texs::Vector{GLuint}
	image_fbos::Vector{GLuint}
	draw_buffers::Vector{GLenum}
	data::Vector{GLubyte}

	function GLData(width_height::Tuple{Int,Int}, mesh_counts::AbstractVector=[], point_counts::AbstractVector=[],
		             vert_shader::String=vert_shader_default,
						 frag_shader::String=frag_shader_default)

		gl = new(0, Vector{GLBuffers}(), Vector{GLBuffers}())

		for m in mesh_counts
			push!(gl.mesh_buffers, GLBuffers(3*m))
		end

		for n in point_counts
			push!(gl.point_buffers, GLBuffers(n))
		end

		gl.image_fbos = Array{GLuint,1}(undef,1)
		gl.image_texs = Array{GLuint,1}(undef,2)

		# Bind offscreen framebuffer to current output
		glGenFramebuffers(1,pointer(gl.image_fbos))
		#gl.image_fbos[1] = glGenFramebuffer()
		glBindFramebuffer(GL_FRAMEBUFFER, gl.image_fbos[1])

		# Color buffer target
		##glGenTextures(1,pointer(gl.image_texs,1))
		gl.image_texs[1] = glGenTexture()
		glBindTexture(GL_TEXTURE_2D, gl.image_texs[1])
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_height..., 0, GL_RGB, GL_UNSIGNED_BYTE, ptr_offset(0))
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl.image_texs[1], 0)

		# Depth buffer target
		##glGenTextures(1,pointer(gl.image_texs,2))
		gl.image_texs[2] = glGenTexture()
		glBindTexture(GL_TEXTURE_2D, gl.image_texs[2])
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width_height..., 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, ptr_offset(0))
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gl.image_texs[2], 0)

		# Location 0 in fragment shader output 
		gl.draw_buffers = Array{GLenum,1}(undef,1)
		gl.draw_buffers[1] = GL_COLOR_ATTACHMENT0
		glDrawBuffers(1,pointer(gl.draw_buffers))

		if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE 
			error("GLData: Incomplete framebuffer")
		end

		gl.data = Array{GLubyte,1}(undef, prod(width_height)*4)
		
		compile!(gl, vert_shader, frag_shader)
		return gl
	end
end

""" High-level viewing geometry and state variables.
"""
mutable struct ViewData
	# Interface
	width::Int
   height::Int
	select::Tuple{Int,Int}
   window::GLFW.Window
	arc_press::SVector{3,<:Number}
	arc_drag::Bool
   mode::Render
   opaque::Bool

	# Geometry
	clip::Tuple{Float64,Float64}
	fov::Float64
	location::SVector{3,<:Number}
	target::SVector{3,<:Number}
	rotation::Matrix{Float64}
	rotation_pre::Matrix{Float64}

	function ViewData(image_size::Tuple{Int,Int}, select::Tuple{Int,Int}=(1,1); mode=colour)

		view = new(image_size[1], image_size[2], select)
		GLFW.Init()
		GLFW.WindowHint(GLFW.SAMPLES, 4)
      GLFW.WindowHint(GLFW.OPENGL_DEBUG_CONTEXT,GL_TRUE)
      GLFW.WindowHint(GLFW.VISIBLE, false)
      view.window = GLFW.CreateWindow(view.width, view.height, "Rendering meshes $(join(select,":"))")
      GLFW.MakeContextCurrent(view.window)
      glViewport(0,0,view.width,view.height)

		# Viewing state
		view.mode = mode
      view.opaque = true
		view.arc_press = @SVector[0.0, 0.0, 0.0]
		view.arc_drag = false

		return view
	end
end

function bounding_box(vertex_arrays)
	vec(extrema(reduce(hcat,stack.(vertex_arrays)), dims=2))
end


mutable struct Renderer <: AbstractRenderer

	view::ViewData
	gl::GLData
	available::NamedTuple{(:colour,:texture,:points),Tuple{Bool,Bool,Bool}}

   @doc """
       Renderer(image_size::Tuple{Int,Int},
                faces::AbstractVector{<:AbstractVector{<:SVector{3,<:Integer}}},
                vertices::AbstractVector{<:AbstractVector{<:SVector{3,<:Real}}},
                normals::AbstractVector{<:AbstractVector{<:SVector{3,<:Real}}};
                texmaps::AbstractVector=[],
                teximgs::AbstractVector=[],
                colours::AbstractVector=[],
                pointclouds::AbstractVector=[],
                centre::Bool=true,
                scale::AbstractVector=[],
                fov::Float64=60.0,
                clip::Tuple=(),
                location::AbstractVector=[],
                target::AbstractVector=[0.0,0.0,0.0],
                backdrop::AbstractVector=[211,215,207]/255)

   Construct a `Renderer`, using the default shaders `src/Vert.glsl` and `src/Frag.glsl`.
   Each vector in `faces` contains the SVector{3,Int} indices for an individual mesh, 
   with reference to the corresponding vertex attributes in `normals`, and optionally `texmaps` 
   and/or `colours`; for example the `faces` argument [`F1`,`F2`] would index vertices [`V1`,`V2`]
   and normals [`N1`,`N2`] of two meshes. In the case of a single object, the surrounding brackets
   are not needed.

   Texture coordinates can be supplied in `texmaps`, with corresponding images in `teximgs`. 

   If RGB `colours` are provided, and length(`colours`)==length(`vertices`) then per-vertex 
   shading can also be applied. Alternatively, if length(`colours`)==length(`faces`) then flat 
   per-face colouring is assumed. The RGB `backdrop` colour may also be specified.
   
   The optional `pointclouds` vector may contain a set of point clouds, associated with the 
   corresponding meshes (in the mandatory arguments).

   If `centre`=true then the meshes are centred on the midpoint of the collective bounding box,
   with an optional overall `scale` applied to the vertices.

   Viewing parameters are set by `fov` (degrees) and `clip`=(near,far). The default camera 
   `location` is [0,0,6`r`], where `r` is the maximum axis-aligned radius of the collective 
   bounding box. The default `target` of the camera is the origin [0,0,0].

   Alternative renderers, using different shaders, can be defined as subtypes of `AbstractRenderer`,
   by making appropriate use of the `buffers!()` function in the constructor.

   Rendering is performed by calling `(AbstractRenderer)()`, as shown below.

   # Examples
       # Render two meshes
       rend = Renderer((w,h), [F1,F2], [V1,V2], [N1,N2])
       rend()
       # Offscreen version
       rend("capture.png")

   # Keyboard controls
   - `Tab`: Show next mesh
   - `Backspace`: Show previous mesh
   - `Shift`+`Tab`: Add next mesh
   - `Shift`+`Backspace`: Remove previous mesh
   - `c`, `t`, `d`, `p`: Render colour, texture, depth, points (where available)
   - `o`: Render colour at 50% opacity
   - `s`: Save image to `meshrender.png` in current directory.
   - `Esc`: Quit
	"""
   function Renderer(image_size::Tuple{Int,Int},
                     faces::AbstractVector{<:AbstractVector{<:SVector{3,<:Integer}}},
                     vertices::AbstractVector{<:AbstractVector{<:SVector{3,<:Real}}},
                     normals::AbstractVector{<:AbstractVector{<:SVector{3,<:Real}}};
                     texmaps::AbstractVector=[],
                     teximgs::AbstractVector=[],
                     colours::AbstractVector=[],
                     pointclouds::AbstractVector=[],
                     centre::Bool=true,
                     scales::AbstractVector=[],
                     fov::Float64=60.0,
                     clip::Tuple=(),
                     location::AbstractVector=[],
                     target::AbstractVector=[0.0,0.0,0.0],
                     backdrop::AbstractVector=[211,215,207]/255)

		vsh = pkgdir(@__MODULE__, "src", "Vert.glsl")
		fsh = pkgdir(@__MODULE__, "src", "Frag.glsl")

		# Set sensible default viewing parameters
		extents = bounding_box(vertices)
		midpoint = SVector{3}(mean.(extents))
		# Camera distance of 6 * (max extents)/2 along +Z axis
		zcam = 6.0 * 0.5*maximum(abs.(Iterators.flatten(extents)))
		location = isempty(location) ? [0.0, 0.0, zcam] : location
		clip = isempty(clip) ? (0.1*zcam, 5.0*zcam) : clip
		if centre
			vertices = map((s,V) -> s*(V.-[midpoint]), 
			               isempty(scales) ? fill(1.0,length(vertices)) : scales,
								vertices)
		end

		# Handle per-face rendering
		if length(faces) == length(colours)
			@assert length(faces) == length(normals) "Per-face rendering requires face normals"
			# Expand/duplicate vertices and attributes  
			vertices = map((F,V) -> V[reduce(vcat,F)], faces, vertices)
			normals = repeat.(normals,inner=3)
			colours = repeat.(colours,inner=3)
			faces = map(V -> 1:length(V), vertices)
		end

		# Set initial rendering mode
		mode = isempty(teximgs) ? colour : texture

		# Allocate ViewData and GLData objects 
		rend = new(ViewData(image_size, (1,length(faces)); mode),
		           GLData(image_size, length.(faces), length.(pointclouds), vsh, fsh), 
					  .!isempty.((colours,teximgs,pointclouds)))

		# Initialize GLData objects
		buffers!(rend.gl.mesh_buffers, [(0,vertices), (1,normals), (2,texmaps), (3,colours)];
		         faces, teximgs)
		buffers!(rend.gl.point_buffers, [(4,pointclouds)])

		# Set viewing parameters
		viewing!(rend; clip, fov, location, target)
		options!(rend; backdrop)
		gl_check()
      return rend
   end
end

"""
    Renderer(image_size::Tuple{Int,Int},
             faces::AbstractVector{<:SVector{3,<:Integer}},
             vertices::AbstractVector{<:SVector{3,<:Real}},
             normals::AbstractVector{<:SVector{3,<:Real}};
             texmaps::AbstractVector=[],
             teximgs::AbstractMatrix=[],
             colours::AbstractVector=[],
             pointclouds::AbstractVector=[], etc...)

Handle the case of a single mesh object, without having to use [F], [V], [N], etc.
"""
function Renderer(image_size::Tuple{Int,Int},
                  faces::AbstractVector{<:SVector{3,<:Integer}},
                  vertices::AbstractVector{<:SVector{3,<:Real}},
                  normals::AbstractVector{<:SVector{3,<:Real}};
						texmaps::AbstractVector=[],
						teximgs::AbstractMatrix=[],
						colours::AbstractVector=[],
						pointclouds::AbstractVector=[], etc...)

	texmaps = isempty(texmaps) ? [] : [texmaps]
	teximgs = isempty(teximgs) ? [] : [teximgs]
	colours = isempty(colours) ? [] : [colours]
	pointclouds = isempty(pointclouds) ? [] : [pointclouds]
	Renderer(image_size, [faces], [vertices], [normals];
	         texmaps, teximgs, colours, pointclouds, etc...)
end

""" Compile and link shaders.
"""
function compile!(gl::GLData, vert_shader::String, frag_shader::String)
   vsh = createShader(read(vert_shader,String), GL_VERTEX_SHADER)
   fsh = createShader(read(frag_shader,String), GL_FRAGMENT_SHADER)
   gl.program = createShaderProgram(vsh,fsh)
   glUseProgram(gl.program)
end

""" Set default options.
"""
function options!(rend::AbstractRenderer; backdrop::AbstractVector)

   glClearColor(backdrop..., 1.0)
   glEnable(GL_BLEND)
   glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
	glEnable(GL_MULTISAMPLE)
   glEnable(GL_DEPTH_TEST)
   glDepthFunc(GL_LESS)
	glEnable(GL_TEXTURE_2D)
	glEnable(GL_CULL_FACE)
   glEnable(GL_PROGRAM_POINT_SIZE)
end

""" Set the viewing parameters.
"""
function viewing!(rend::AbstractRenderer;
	               clip::Tuple=rend.view.clip,
						fov::Float64=rend.view.fov,
	               location::AbstractVector=rend.view.location,
						target::AbstractVector=rend.view.target)

   rend.view.clip = clip
   rend.view.fov = fov
   rend.view.location = location
   rend.view.target = target
	rend.view.rotation = I(3)
	rend.view.rotation_pre = I(3)

   glUniform1f(glGetUniformLocation(rend.gl.program,"near"), GLfloat(rend.view.clip[1]))
   glUniform1f(glGetUniformLocation(rend.gl.program,"far"), GLfloat(rend.view.clip[2]))
   P = perspective(rend.view.clip, rend.view.fov, 1.0)
   glUniformMatrix4fv(glGetUniformLocation(rend.gl.program,"projection"), 1, false, gl_vec(P))
end

""" Load data buffers for shaders.
"""
function buffers!(bufs::Vector{GLBuffers}, attributes::Vector{Tuple{Int,T}}; 
                  faces=[], teximgs=[]) where {T<:Array}

	# Remove any empty attribute arrays
	attributes = filter(a -> !isempty(last(a)), attributes)

	for i in 1:length(bufs)

		# Attributes of i-th object
		attribs = map(a -> last(a)[i], attributes)
		# Vec of stacked attribute matrices (vertex data in columns)
		data = gl_vec(vcat(stack.(attribs)...))
		# Transfer the data
		glBindVertexArray(bufs[i].vao)
		glBindBuffer(GL_ARRAY_BUFFER, bufs[i].vbo)
		glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
		sizes = first.(size.(first.(attribs)))
		stride = GLsizei.(sum(sizes) * sizeof(GLfloat))
		offsets = [0; cumsum(sizes)]
		println("Attribute sizes: $(sizes), offsets: $(offsets[1:end-1]), stride: $(stride).")
		for j in 1:length(attributes)
			glVertexAttribPointer(first(attributes[j]), sizes[j], GL_FLOAT, GL_FALSE,
			                      stride, ptr_offset(offsets[j]))
			glEnableVertexAttribArray(first(attributes[j]))
		end

		# Indices
		if isempty(faces)
			# Default consecutive indexing
			indices = gl_vec(collect(0 : bufs[i].n-1), GLuint)
		else
			# Flatten the face indices
			indices = gl_vec(reduce(vcat,faces[i]) .- 1, GLuint)
		end
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs[i].ibo)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW)

		# Textures
		if !isempty(teximgs)
			# Flip and cast image.
			img = reinterpret(GLubyte, transpose(teximgs[i][end:-1:1,:]))
			num_channels = length(teximgs[i][1])
			if num_channels == 3
				fmt = GL_RGB
			elseif num_channels == 4
				fmt = GL_RGBA
			else
				error("Unhandled colour type (not RGB or RGBA).")
			end
			glActiveTexture(GL_TEXTURE0 + i-1)
			glBindTexture(GL_TEXTURE_2D, bufs[i].tex)
			glTexImage2D(GL_TEXTURE_2D, 0, fmt, size(teximgs[i])..., 0, fmt, GL_UNSIGNED_BYTE, img)
			glGenerateMipmap(GL_TEXTURE_2D);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		end
	end
end

""" Render one frame.
"""
function render(rend::AbstractRenderer)
   if rend.view.mode == colour || rend.view.mode == depth || rend.view.mode == texture
		for k in range(rend.view.select...)
			glUniform1i(glGetUniformLocation(rend.gl.program,"teximg"), k-1)
			glBindVertexArray(rend.gl.mesh_buffers[k].vao)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rend.gl.mesh_buffers[k].ibo)
			glDrawElements(GL_TRIANGLES, rend.gl.mesh_buffers[k].n, GL_UNSIGNED_INT, ptr_offset(0))
		end
	elseif rend.view.mode == points
		for k in range(rend.view.select...)
			glBindVertexArray(rend.gl.point_buffers[k].vao)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rend.gl.point_buffers[k].ibo)
			glDrawElements(GL_POINTS, rend.gl.point_buffers[k].n, GL_UNSIGNED_INT, ptr_offset(0))
		end
	end
end

""" Update viewing geometry.
"""
function update!(rend::AbstractRenderer) 
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
   M = modelview(rend.view.location, rend.view.target, rotation=rend.view.rotation)
   glUniformMatrix4fv(glGetUniformLocation(rend.gl.program,"modelview"), 1, false, gl_vec(M))
end

""" Set interface callbacks and start the Renderer.
"""
function (rend::AbstractRenderer)(; opts...)

	# Optional update of viewing parameters 
	viewing!(rend; opts...)

	# Initalize as opaque
   glUniform1f(glGetUniformLocation(rend.gl.program,"opacity"), GLfloat(1.0))

	# Show the hidden window
	GLFW.ShowWindow(rend.view.window)

	# Key controls
   GLFW.SetKeyCallback(rend.view.window,
		(window::GLFW.Window, button::GLFW.Key, code::Int32, action::GLFW.Action, mods::Int32) ->
		begin
			if button == GLFW.KEY_I && action == GLFW.PRESS
				rend("meshrender.png"; opts...)
			
			elseif button == GLFW.KEY_C && action == GLFW.PRESS && rend.available.colour
				rend.view.mode = colour
			
			elseif button == GLFW.KEY_T && action == GLFW.PRESS && rend.available.texture
				rend.view.mode = texture

			elseif button == GLFW.KEY_D && action == GLFW.PRESS
				rend.view.mode = depth
			
			elseif button == GLFW.KEY_P && action == GLFW.PRESS && rend.available.points
				rend.view.mode = points
			
			elseif button == GLFW.KEY_O && action == GLFW.PRESS
				rend.view.opaque = !rend.view.opaque
				glUniform1f(glGetUniformLocation(rend.gl.program,"opacity"), GLfloat((1.0+rend.view.opaque)/2.0))

			elseif mods == GLFW.MOD_SHIFT && button == GLFW.KEY_TAB && action == GLFW.PRESS
				# Add subsequent mesh
				rend.view.select = (rend.view.select[1], min(rend.view.select[2]+1, length(rend.gl.mesh_buffers)))
		
			elseif mods == GLFW.MOD_SHIFT && button == GLFW.KEY_BACKSPACE && action == GLFW.PRESS
				# Remove last mesh
				rend.view.select = (rend.view.select[1], max(rend.view.select[2]-1, rend.view.select[1]))
			
			elseif button == GLFW.KEY_TAB && action == GLFW.PRESS
				# Merge and increment the indices
				k = min(rend.view.select[1]+1, length(rend.gl.mesh_buffers))
				rend.view.select = (k,k)

			elseif button == GLFW.KEY_BACKSPACE && action == GLFW.PRESS
				# Merge and decrement the indices
				k = max(rend.view.select[1]-1, 1)
				rend.view.select = (k,k)
			
			elseif button == GLFW.KEY_ESCAPE
				GLFW.SetWindowShouldClose(rend.view.window,true)
			end

			glUniform1i(glGetUniformLocation(rend.gl.program,"render_mode"), GLint(rend.view.mode))
			GLFW.SetWindowTitle(rend.view.window, "Rendering meshes $(join(rend.view.select,":"))")
   	end)

	# Arcball button controls
	GLFW.SetMouseButtonCallback(rend.view.window,
		(window::GLFW.Window, button::GLFW.MouseButton, action::GLFW.Action, mods::Int32) ->
		begin
			if button == GLFW.MOUSE_BUTTON_LEFT
				if action == GLFW.PRESS
					# Store rotation state & start new arc
					rend.view.rotation_pre = rend.view.rotation
					rend.view.arc_press = arcball_vector(SVector{2,<:Number}(GLFW.GetWindowSize(window)...), 
					                                SVector{2,<:Number}(GLFW.GetCursorPos(window)...))
					rend.view.arc_drag = true
				elseif action == GLFW.RELEASE
					rend.view.arc_drag = false
				end
			end
		end)

	# Arcball drag controls
	GLFW.SetCursorPosCallback(rend.view.window,
		(window::GLFW.Window, x::Float64, y::Float64) ->
		begin
			if rend.view.arc_drag
				v = arcball_vector(SVector{2,<:Number}(GLFW.GetWindowSize(window)...), SVector{2,<:Number}(x,y))
				t = angle(rend.view.arc_press, v)
				n = cross(rend.view.arc_press, v)
				# Compose doubled differential rotation onto previous state
				rend.view.rotation = rotation(2.0*t,n) * rend.view.rotation_pre
			end
		end)
	
	# Scrollwheel zoom control
	GLFW.SetScrollCallback(rend.view.window,
		(window::GLFW.Window, x::Float64, y::Float64) ->
		begin
			rend.view.location = rend.view.location + [0.0, 0.0, 0.1*rend.view.location[3]*y]
		end)

	# Window resizing/reshaping
  	GLFW.SetWindowSizeCallback(rend.view.window,
	   (window::GLFW.Window, w::Int32, h::Int32) ->
		begin
			P = perspective(rend.view.clip, rend.view.fov, w/h)
			glUniformMatrix4fv(glGetUniformLocation(rend.gl.program,"projection"), 1, false, gl_vec(P))
			glViewport(0,0,w,h)
			rend.view.arc_drag = false
		end)

	# Set shader options
   glUniform1i(glGetUniformLocation(rend.gl.program,"render_mode"), GLint(rend.view.mode))
	# Bind location 0 in fragments shader output to display
   glBindFramebuffer(GL_FRAMEBUFFER,0)

   while !GLFW.WindowShouldClose(rend.view.window)
      update!(rend)
      render(rend)
      GLFW.SwapBuffers(rend.view.window)
      GLFW.PollEvents()
   end
   GLFW.DestroyWindow(rend.view.window)
	GLFW.Terminate()
end


function (rend::AbstractRenderer)(file::String; image_function=(depth)->depth, opts...)

	# Optional update of viewing parameters 
	viewing!(rend; opts...)

   glUniform1i(glGetUniformLocation(rend.gl.program,"render_mode"), GLint(rend.view.mode))
   glUniform1f(glGetUniformLocation(rend.gl.program,"opacity"), GLfloat(1.0))

	# Set texture as target for drawing via framebuffer object
   glBindFramebuffer(GL_FRAMEBUFFER, rend.gl.image_fbos[1])

	if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE 
		error("Incomplete framebuffer")
	end

   # Re-render in the current mode
   update!(rend)
   render(rend)
	# Reset to display
	glBindFramebuffer(GL_FRAMEBUFFER, 0)

	# Read from texture
	glBindTexture(GL_TEXTURE_2D, rend.gl.image_texs[1])
   glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pointer(rend.gl.data))

   println("depth range: $(extrema(Float64.(rend.gl.data)))")

   image = colorview(RGBA, normedview(reshape(rend.gl.data, (4,rend.view.width,rend.view.height))))

   save(file, transpose(image[:,end:-1:1]))
end

function simulate_depthcam(depth)
   tmp = imresize(depth, (400,400), method=BSpline(Constant()))
   imresize(tmp, (1600,1600), method=BSpline(Constant()))
end

"""
    render_objs(objnames::AbstractVector; texnames::AbstractVector=[])

Load a list of OBJ mesh files, with corresponding textures, and render them using default options.
An intermediate `GeometryBasics` representation is used, in order to decompose the meshes.
"""
function render_objs(objnames::AbstractVector=["spot/model.obj", "banana/model.obj"], 
                     texnames::AbstractVector=["spot/texture.png","banana/texture.png"])

	meshes = GeometryBasics.expand_faceviews.(GeometryBasics.uv_normal_mesh.(load.(objnames)))
	faces = pmap(M -> SVector{3,UInt32}.(GeometryBasics.faces(M)), meshes)
	vertices = pmap(M -> SVector{3,Float32}.(GeometryBasics.coordinates(M)), meshes)
	normals = pmap(M -> SVector{3,Float32}.(GeometryBasics.normals(M)), meshes)
	texmaps = pmap(M -> SVector{2,Float32}.(GeometryBasics.values(GeometryBasics.texturecoordinates(M))), meshes)
	teximgs = load.(texnames)
	rend = MeshRender.Renderer((1200,1200), faces, vertices, normals; texmaps, teximgs)
	rend()
end


end

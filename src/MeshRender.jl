module MeshRender

using StaticArrays, LinearAlgebra, StatsBase, GLFW, ModernGL, FileIO, VisionGeometry,
Images, Colors, ImageTransformations, Interpolations

import GeometryBasics

export Renderer, FlatRenderer, __FlatRenderer, viewing!

# Auxiliary files
include(pkgdir(ModernGL, "test", "util.jl"))

""" Construct OpenGL camera matrix from [near,far] limits (unsigned),
    field of view (degrees), and aspect ratio.
"""
function perspective(clip::Tuple{Float64,Float64}, fov_v_deg::Float64, aspect::Float64=1.0)
   # Homogeneous perspective
   f = 1.0/tan(radians(fov_v_deg)/2.0)
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
function modelview(cam::GVector{3}, at::GVector{3}; up::AbstractVector=[0.0; 1.0; 0.0],
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

function gl_image(window::GLFW.Window)
   fb_size = GLFW.GetFramebufferSize(window)
   gl_data = gl_vec(Array{UInt8,1}(undef,prod(fb_size)*4), GLubyte)
   glPixelStorei(GL_PACK_ALIGNMENT, 4)
   glReadBuffer(GL_FRONT)
   glReadPixels(0, 0, GLint(fb_size[1]), GLint(fb_size[2]), 
                GL_RGBA, GL_UNSIGNED_BYTE, pointer(gl_data))
   colorview(RGBA, normedview(reshape(gl_data,(4,fb_size[1],fb_size[2]))))
   imshow(img, axes=(2,1), flipy=true)
end

""" Compute z-component of the generalized arcball vector.
"""
function arcball_depth(v::GVector{2}, r1::Float64=1.0)
	r_sqr = v[1]^2 + v[2]^2
	(r_sqr <= 0.5*r1^2) ? sqrt(r1^2-r_sqr) : (0.5*r1^2)/sqrt(r_sqr)
end

""" Compute the generalized arcball vector. 
"""
function arcball_vector(image_size::GVector{2}, cursor_pos::GVector{2})
	# Center and radial 2D vector
	c = (image_size .- 1.0) ./ 2.0
	q = (cursor_pos .- c) ./ (min(image_size...)-1.0)
	# Radial 3D vector
	GVector{3}(q[1], -q[2], arcball_depth(q))
end

"Initialize GL vector from concatenated array"
function gl_vec(M::AbstractArray, gl_type::DataType=GLfloat)
   Array{gl_type,1}(vec(M))
end

"Byte offset as multiple of type size"
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
		glBindFramebuffer(GL_FRAMEBUFFER, gl.image_fbos[1])

		# Color buffer target
		glGenTextures(1,pointer(gl.image_texs,1))
		glBindTexture(GL_TEXTURE_2D, gl.image_texs[1])
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_height..., 0, GL_RGB, GL_UNSIGNED_BYTE, ptr_offset(0))
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl.image_texs[1], 0)

		# Depth buffer target
		glGenTextures(1,pointer(gl.image_texs,2))
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

mutable struct ViewData
	# Interface
	width::Int
   height::Int
	select::Tuple{Int,Int}
   window::GLFW.Window
	arc_press::GVector{3}
	arc_drag::Bool
   mode::Render
   opaque::Bool

	# Geometry
	scale::Float64
	clip::Tuple{Float64,Float64}
	fov::Float64
	location::GVector{3}
	target::GVector{3}
	rotation::Matrix{Float64}
	rotation_pre::Matrix{Float64}

	function ViewData(image_size::Tuple{Int,Int}, select::Tuple{Int,Int}=(1,1); visible=true, mode=colour)

		view = new(image_size[1], image_size[2], select)
		GLFW.Init()
		GLFW.WindowHint(GLFW.SAMPLES, 4)
      GLFW.WindowHint(GLFW.OPENGL_DEBUG_CONTEXT,GL_TRUE)
      GLFW.WindowHint(GLFW.VISIBLE, visible)
      view.window = GLFW.CreateWindow(view.width, view.height, "Rendering mesh range $(view.select)")
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
                faces::Vector{<:Vector{<:SVector{3,<:Integer}}},
                vertices::Vector{<:Vector{<:SVector{3,<:Real}}},
                normals::Vector{<:Vector{<:SVector{3,<:Real}}};
                texmaps::AbstractVector=[],
                teximgs::AbstractVector=[],
                colours::AbstractVector=[],
                points::AbstractVector=[],
                centre::Bool=true,
                scale::Float64=1.0,
                fov::Float64=60.0,
                clip::Tuple=(),
                location::AbstractVector=[],
                target::AbstractVector=[0.0,0.0,0.0],
                visible=true)

   Construct a `Renderer`, using the default shaders `Vert.glsl` and `Frag.glsl`.
   Each vector in `faces` contains the index-triples for an individual mesh, referring 
   to the corresponding vertex attributes in `normals`, and optionally `texmaps` and/or
   `colours`.

   Alternatively, if length(`colours`)==length(`faces`) then flat per-face colouring is assumed.
   
   The optional `points` vector may contain a set of pointclouds, associated with the 
   corresponding meshes in the mandatory arguments.

   # Examples
       # Multiple meshes
       rend = Renderer((w,h), [F1,F2], [V1,V2], [N1,N2])
       rend()
       # Offscreen rendering of a single mesh
       rend = Renderer((w,h), F, V, N, visible=false)
       rend("capture.png")
   
   The default camera `location` is [0,0,6`r`], where `r` is the maximum axis-aligned
   radius of the collective bounding box.

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
                     faces::Vector{<:Vector{<:SVector{3,<:Integer}}},
                     vertices::Vector{<:Vector{<:SVector{3,<:Real}}},
                     normals::Vector{<:Vector{<:SVector{3,<:Real}}};
                     texmaps::AbstractVector=[],
                     teximgs::AbstractVector=[],
                     colours::AbstractVector=[],
                     points::AbstractVector=[],
                     centre::Bool=true,
                     scale::Float64=1.0,
                     fov::Float64=60.0,
                     clip::Tuple=(),
                     location::AbstractVector=[],
                     target::AbstractVector=[0.0,0.0,0.0],
                     visible=true)

		vsh = pkgdir(@__MODULE__, "src", "Vert.glsl")
		fsh = pkgdir(@__MODULE__, "src", "Frag.glsl")

		extents = bounding_box(vertices)
		midpoint = SVector{3}(mean.(extents))
		# Default viewing parameters, based on camera at distance of 3 * (max extents/2).
		zcam = 6.0 * scale * 0.5*maximum(abs.(Iterators.flatten(extents)))
		location = isempty(location) ? [0.0, 0.0, zcam] : location
		clip = isempty(clip) ? (0.1*zcam, 5.0*zcam) : clip
		if centre || scale != 1.0
			vertices = map(V -> scale*(V.-[midpoint]), vertices)
		end

		if length(faces) == length(colours)
			# Expand/duplicate vertices and attributes  
			vertices = map((F,V) -> V[reduce(vcat,F)], faces, vertices)
			normals = repeat.(normals,inner=3)
			colours = repeat.(colours,inner=3)
			faces = map(V -> 1:length(V), vertices)
		end

		mode = isempty(teximgs) ? colour : texture

		rend = new(ViewData(image_size; mode),
		           GLData(image_size, length.(faces), length.(points), vsh, fsh), 
					  .!isempty.((colours,teximgs,points)))

		buffers!(rend.gl.mesh_buffers, [(0,vertices), (1,normals), (2,texmaps), (3,colours)];
		         faces, teximgs)
		buffers!(rend.gl.point_buffers, [(4,points)])

		viewing!(rend; scale, clip, fov, location, target)
		gl_check()
      return rend
   end
end

mutable struct FlatRenderer <: AbstractRenderer

	view::ViewData
	gl::GLData

   @doc """
   Construct a `FlatRenderer`.
	"""
   function FlatRenderer(image_size::Tuple{Int,Int}, faces::Vector{Vector{SVector{3,Ind}}}, vertices::Vector{GVectors{3}};
							    normals::Vector{Vector{SVector{3,Coord}}},
							    colours::Vector{GVectors{3}}=nothing,
							    points::Vector{GVectors{3}}=nothing,
		                   scale::Float64=1.0, clip::Tuple{Float64,Float64}=(0.1,30.0), fov::Float64=60.0,
							    location::AbstractVector=[0.0,0.0,5.0], target::AbstractVector=[0.0,0.0,0.0],
							    visible=true) where {Ind <: Integer, Coord <: Number}


		vsh = pkgdir(@__MODULE__, "src", "FlatVert.glsl")
		fsh = pkgdir(@__MODULE__, "src", "FlatFrag.glsl")

      rend = new(ViewData(image_size),
                 GLData(image_size, length.(faces), length.(points), vsh, fsh))
		
		# Expand/duplicate vertices and attributes  
		all_vertices = map((F,V) -> V[reduce(vcat,F)], faces, vertices)
		all_normals = repeat.(normals,inner=3)
		all_colours = repeat.(colours,inner=3)

		buffers!(rend.gl.mesh_buffers, [(0,all_vertices),(1,all_normals),(2,all_colours)])
		buffers!(rend.gl.point_buffers, [(3,points)])
		
		viewing!(rend; scale, clip, fov, location, target)
      return rend
   end
end


""" Compile and link shaders.
"""
function compile!(gl::GLData, vert_shader::String, frag_shader::String)
   vsh = createShader(read(vert_shader,String), GL_VERTEX_SHADER)
   fsh = createShader(read(frag_shader,String), GL_FRAGMENT_SHADER)
   gl.program = createShaderProgram(vsh,fsh)
   glUseProgram(gl.program)
	gl_check()
end

""" Set default options.
"""
function options!(rend::AbstractRenderer; 
	               background::AbstractVector=[211,215,207]/255, 
	               blend::Tuple{UInt32,UInt32}=(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA))

   glClearColor(background..., 1.0)
   glEnable(GL_BLEND)
   glBlendFunc(blend...)
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
	               scale::Float64, clip::Tuple{Float64,Float64}, fov::Float64,
	               location::AbstractVector, target::AbstractVector)

   rend.view.scale = scale
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
function buffers!(bufs::Vector{GLBuffers}, attributes::Vector{Tuple{Int,T}}; faces=[], teximgs=[]) where {T<:Array}

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
		println("sizes, offsets, stride: $(sizes), $(offsets[1:end-1]), $(stride).")
		for j in 1:length(attributes)
			glVertexAttribPointer(first(attributes[j]), sizes[j], GL_FLOAT, GL_FALSE, stride, ptr_offset(offsets[j]))
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
			#glDrawArrays(GL_TRIANGLES, 0, rend.gl.mesh_buffers[k].n)
		end
	elseif rend.view.mode == points
		for k in range(rend.view.select...)
			glBindVertexArray(rend.gl.point_buffers[k].vao)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rend.gl.point_buffers[k].ibo)
			glDrawElements(GL_POINTS, rend.gl.point_buffers[k].n, GL_UNSIGNED_INT, ptr_offset(0))
			#glDrawArrays(GL_POINTS, 0, rend.gl.point_buffers[k].n)
		end
	end
end

""" Update viewing geometry.
"""
function update!(rend::AbstractRenderer) 
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
   M = modelview(rend.view.location, rend.view.target, rotation=rend.view.rotation, scale=rend.view.scale)
   glUniformMatrix4fv(glGetUniformLocation(rend.gl.program,"modelview"), 1, false, gl_vec(M))
end

""" Set interface callbacks and start the Renderer.
"""
function (rend::AbstractRenderer)(; opts...)

	options!(rend; opts...)

	# Initalize as opaque
   glUniform1f(glGetUniformLocation(rend.gl.program,"opacity"), GLfloat(1.0))

	# Key controls
   GLFW.SetKeyCallback(rend.view.window,
		(window::GLFW.Window, button::GLFW.Key, code::Int32, action::GLFW.Action, mods::Int32) ->
		begin
			if button == GLFW.KEY_ESCAPE
				GLFW.SetWindowShouldClose(rend.view.window,true)
			
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
			
			elseif button == GLFW.KEY_I && action == GLFW.PRESS
				#gl_check()
				rend("meshrender.png"; opts...)
			end
			glUniform1i(glGetUniformLocation(rend.gl.program,"render_mode"), GLint(rend.view.mode))
			GLFW.SetWindowTitle(rend.view.window, "Rendering mesh range $(rend.view.select)")
   	end)

	# Arcball button controls
	GLFW.SetMouseButtonCallback(rend.view.window,
		(window::GLFW.Window, button::GLFW.MouseButton, action::GLFW.Action, mods::Int32) ->
		begin
			if button == GLFW.MOUSE_BUTTON_LEFT
				if action == GLFW.PRESS
					# Store rotation state & start new arc
					rend.view.rotation_pre = rend.view.rotation
					rend.view.arc_press = arcball_vector(GVector{2}(GLFW.GetWindowSize(window)...), 
					                                GVector{2}(GLFW.GetCursorPos(window)...))
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
				v = arcball_vector(GVector{2}(GLFW.GetWindowSize(window)...), GVector{2}(x,y))
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
			rend.view.location = rend.view.location + [0.0, 0.0, y]
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

display(rend.gl.image_texs)

display(map(b->b.tex, rend.gl.mesh_buffers))

	options!(rend; opts...)

   glUniform1i(glGetUniformLocation(rend.gl.program,"render_mode"), GLint(rend.view.mode))
   glUniform1f(glGetUniformLocation(rend.gl.program,"opacity"), GLfloat(1.0))

	# Set texture as target for drawing via framebuffer object
   glBindFramebuffer(GL_FRAMEBUFFER, rend.gl.image_fbos[1])

	if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE 
		error("Incomplete framebuffer")
	end
	gl_check()

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

##################################

function simulate_depthcam(depth)
   tmp = imresize(depth, (400,400), method=BSpline(Constant()))
   imresize(tmp, (1600,1600), method=BSpline(Constant()))
end

function attrib_matrix(indices::AbstractArray, attribs::AbstractArray)
	# Flatten indices and remove duplicates
	J = unique(reduce(vcat,indices))

	# Length of concatenated arributes 
	m = sum(length.(first.(attribs)))

	# Raw matrix to hold attributes in columns
	A = Array{Float32}(undef, m, maximum(J))

	println("Constructed $(size(A)) attribute array")

	# Assignment
	A[:,J] .= stack(vcat.(map(a->a[J],attribs)...))

	return deepcopy(A)
end

function test_obj()

	#using FileIO, GeometryBasics, VisionGeometry, MeshRender

	obj_name = "spot/model.obj"
	tex_name = "spot/texture.png"

	obj = load(obj_name)
	mesh = GeometryBasics.expand_faceviews(GeometryBasics.uv_normal_mesh(obj))
	F = SVector{3,UInt32}.(GeometryBasics.faces(mesh))
	V = SVector{3,Float32}.(GeometryBasics.coordinates(mesh))
	N = SVector{3,Float32}.(GeometryBasics.normals(mesh))
	TM = SVector{2,Float32}.(GeometryBasics.values(GeometryBasics.texturecoordinates(mesh)))
	TI = load(tex_name)

	obj_name = "banana/model.obj"
	tex_name = "banana/texture.png"

	obj = load(obj_name)
	mesh = GeometryBasics.expand_faceviews(GeometryBasics.uv_normal_mesh(obj))
	F2 = SVector{3,UInt32}.(GeometryBasics.faces(mesh))
	V2 = SVector{3,Float32}.(GeometryBasics.coordinates(mesh))
	N2 = SVector{3,Float32}.(GeometryBasics.normals(mesh))
	TM2 = SVector{2,Float32}.(GeometryBasics.values(GeometryBasics.texturecoordinates(mesh)))
	TI2 = load(tex_name)

   rend = MeshRender.Renderer((1200,1200), [F,F2], [V,V2], [N,N2], texmaps=[TM,TM2], teximgs=[TI,TI2], centre=true, scale=1.0)
	#rend = MeshRender.Renderer((1200,1200), [F,F2], [V,V2], [N,N2])

	rend()
end


end

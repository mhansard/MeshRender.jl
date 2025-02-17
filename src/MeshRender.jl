module MeshRender

using StaticArrays, GLFW, ModernGL, LinearAlgebra, FileIO, VisionGeometry,
Images, ImageTransformations, Interpolations

export FlatRenderer, options!, viewing!

# Auxiliary files
include(pkgdir(ModernGL, "test", "util.jl"))
const vert_shader_default = pkgdir(@__MODULE__, "src", "MeshRenderVert.glsl")
const frag_shader_default = pkgdir(@__MODULE__, "src", "MeshRenderFrag.glsl")

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
function arcball_vector(window_size::GVector{2}, cursor_pos::GVector{2})
	# Center and radial 2D vector
	c = (window_size .- 1.0) ./ 2.0
	q = (cursor_pos .- c) ./ (min(window_size...)-1.0)
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
		error("Failed: $(GLENUM(err).name)" * msg)
	end
end

abstract type AbstractRenderer end 

@enum Render colour=1 depth=2 points=3 texture=4

struct GLBuffers
	vao::GLuint
	vbo::GLuint
	ibo::GLuint
	n::GLuint
	function GLBuffers(n::Int)
		new(glGenVertexArray(), glGenBuffer(), glGenBuffer(), n)
	end
end

mutable struct GLData

	program::GLuint
	mesh_buffers::Vector{GLBuffers}
	point_buffers::Vector{GLBuffers}

	image_tx::Vector{GLuint}
	image_fb::Vector{GLuint}
	draw_bf::Vector{GLenum}
	data::Vector{GLubyte}

	function GLData(width_height::Tuple{Int,Int}, mesh_counts::Vector{Int}, point_counts::Vector{Int},
		             vert_shader::String=vert_shader_default,
						 frag_shader::String=frag_shader_default)

		gl = new(0, Vector{GLBuffers}(), Vector{GLBuffers}())

		for m in mesh_counts
			push!(gl.mesh_buffers, GLBuffers(m))
		end

		for n in point_counts
			push!(gl.point_buffers, GLBuffers(n))
		end

		gl.image_fb = Array{GLuint,1}(undef,1)
		glGenFramebuffers(1,pointer(gl.image_fb))
		glBindFramebuffer(GL_FRAMEBUFFER, gl.image_fb[1])

		gl.image_tx = Array{GLuint,1}(undef,2)

		# Color buffer
		glGenTextures(1,pointer(gl.image_tx,1))
		glBindTexture(GL_TEXTURE_2D, gl.image_tx[1])
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_height..., 0, GL_RGB, GL_UNSIGNED_BYTE, ptr_offset(0))
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl.image_tx[1], 0)

		# Depth buffer
		glGenTextures(1,pointer(gl.image_tx,2))
		glBindTexture(GL_TEXTURE_2D, gl.image_tx[2])
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width_height..., 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, ptr_offset(0))
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gl.image_tx[2], 0)

		gl.draw_bf = Array{GLenum,1}(undef,1)
		gl.draw_bf[1] = GL_COLOR_ATTACHMENT0
		glDrawBuffers(1,pointer(gl.draw_bf))
		status_fb = glCheckFramebufferStatus(GL_FRAMEBUFFER)
		glBindFramebuffer(GL_FRAMEBUFFER, gl.image_fb[1])
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

	function ViewData(window_size::Tuple{Int,Int}, select::Tuple{Int,Int}=(1,1), visible=true)

		view = new(window_size[1], window_size[2], select)
		GLFW.Init()
		GLFW.WindowHint(GLFW.SAMPLES, 4)
      GLFW.WindowHint(GLFW.OPENGL_DEBUG_CONTEXT,GL_TRUE)
      GLFW.WindowHint(GLFW.VISIBLE, visible)
      view.window = GLFW.CreateWindow(view.width, view.height, "Rendering mesh range $(view.select)")
      GLFW.MakeContextCurrent(view.window)
      glViewport(0,0,view.width,view.height)

		# Viewing state
		view.mode = colour
      view.opaque = true
		view.arc_press = @SVector[0.0, 0.0, 0.0]
		view.arc_drag = false

		return view
	end
end

mutable struct FlatRenderer <: AbstractRenderer

	view::ViewData
	gl::GLData

   @doc """
       FlatRenderer(window_size::Tuple{Int,Int}, F::Vector{IVectors{3}}, V::Vector{GVectors{3}},
						  N::Vector{GVectors{3}}=nothing,
						  C::Vector{GVectors{3}}=nothing, 
						  P::Vector{Vector{GVectors{3}}}=nothing;
		              scale::Float64=1.0, clip::Tuple{Float64,Float64}=(0.1,30.0), fov::Float64=60.0,
						  location::AbstractVector=[0.0,0.0,5.0], target::AbstractVector=[0.0,0.0,0.0],
						  visible=true)

   Construct a `FlatRenderer`.
	"""
   function FlatRenderer(window_size::Tuple{Int,Int}, F::Vector{IVectors{3}}, V::Vector{GVectors{3}},
							    N::Vector{GVectors{3}}=nothing,
							    C::Vector{GVectors{3}}=nothing, 
							    P::Vector{Vector{GVectors{3}}}=nothing;
		                   scale::Float64=1.0, clip::Tuple{Float64,Float64}=(0.1,30.0), fov::Float64=60.0,
							    location::AbstractVector=[0.0,0.0,5.0], target::AbstractVector=[0.0,0.0,0.0],
							    visible=true)

		mesh_data = map((v,f,n,c) -> gl_vec(vcat(stack(v[reduce(vcat,f)]),
					                                stack(repeat(n,inner=3)),
					                                stack(repeat(c,inner=3)))), V,F,N,C)

		point_data = map(p -> gl_vec(reduce(hcat, stack.(p))), P)

		vertex_counts = 3*length.(F)
		point_counts = sum.(map(p->length.(p), P))

      rend = new(ViewData(window_size), GLData(window_size, vertex_counts, point_counts))
		
		buffers!(rend.gl.mesh_buffers, [0,1,2], [3,3,3], mesh_data)
		buffers!(rend.gl.point_buffers, [3], [3], point_data)

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
	# Set default options
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
	# Set default options
	options!(rend)
end

""" Load data buffers for shaders.
"""
function buffers!(bufs::Vector{GLBuffers}, locs::Vector{Int}, lens::Vector{Int}, data, faces=nothing)

	# Treat each v/n/c as a 3x1 column, and form a block array from the K meshes: 
	#    V1 V2 ... VK
	#    N1 N2 ... VK
	#    C1 C2 ... VK
	# Then vec the entire array, columnwise.

	for i in 1:length(bufs)
		
		glBindVertexArray(bufs[i].vao)
		glBindBuffer(GL_ARRAY_BUFFER, bufs[i].vbo)
		glBufferData(GL_ARRAY_BUFFER, sizeof(data[i]), data[i], GL_STATIC_DRAW)

		stride = sum(lens) * sizeof(GL_FLOAT)
		offs = cumsum(lens) .- first(lens)

		for j in 1:length(locs)
			glVertexAttribPointer(locs[j], lens[j], GL_FLOAT, GL_FALSE, stride, ptr_offset(offs[j]))
			glEnableVertexAttribArray(locs[j])
		end

		if isnothing(faces)
			indices = gl_vec(collect(0 : bufs[i].n-1), GLuint)
		else
			indices = reduce(vcat, faces[i])
		end
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs[i].ibo)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW)
	end

	gl_check()
end

""" Render one frame.
"""
function render(rend::AbstractRenderer)
   if rend.view.mode == colour || rend.view.mode == depth
		for k in range(rend.view.select...)
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
function (rend::AbstractRenderer)()
	# Initalize as opaque
   glUniform1f(glGetUniformLocation(rend.gl.program,"opacity"), GLfloat(1.0))

	# Key controls
   GLFW.SetKeyCallback(rend.view.window,
		(window::GLFW.Window, button::GLFW.Key, code::Int32, action::GLFW.Action, mods::Int32) ->
		begin
			if button == GLFW.KEY_ESCAPE
				GLFW.SetWindowShouldClose(rend.view.window,true)
			elseif button == GLFW.KEY_C && action == GLFW.PRESS
				rend.view.mode = colour
			elseif button == GLFW.KEY_D && action == GLFW.PRESS
				rend.view.mode = depth
			elseif button == GLFW.KEY_P && action == GLFW.PRESS
				rend.view.mode = points
			elseif button == GLFW.KEY_O && action == GLFW.PRESS
				rend.view.opaque = !rend.view.opaque
				glUniform1f(glGetUniformLocation(rend.gl.program,"opacity"), GLfloat((1.0+rend.view.opaque)/2.0))
			elseif button == GLFW.KEY_LEFT && action == GLFW.PRESS
				# Merge and decrement the indices
				k = max(rend.view.select[1]-1, 1)
				rend.view.select = (k,k)
			elseif button == GLFW.KEY_RIGHT && action == GLFW.PRESS
				# Merge and increment the indices
				k = min(rend.view.select[1]+1, length(rend.gl.mesh_buffers))
				rend.view.select = (k,k)
			elseif button == GLFW.KEY_UP && action == GLFW.PRESS
				# Add subsequent mesh
				rend.view.select = (rend.view.select[1], min(rend.view.select[2]+1, length(rend.gl.mesh_buffers)))
			elseif button == GLFW.KEY_DOWN && action == GLFW.PRESS
				# Remove last mesh
				rend.view.select = (rend.view.select[1], max(rend.view.select[2]-1, rend.view.select[1]))
			elseif button == GLFW.KEY_I && action == GLFW.PRESS
				rend("tmp.png")
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

   glUniform1i(glGetUniformLocation(rend.gl.program,"render_mode"), GLint(rend.view.mode))
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


function (rend::AbstractRenderer)(file::String, image_function=(depth)->depth)
	
   glUniform1i(glGetUniformLocation(rend.gl.program,"render_mode"), GLint(rend.view.mode))
   glUniform1f(glGetUniformLocation(rend.gl.program,"opacity"), GLfloat(1.0))

	# Set first texture as target
   glBindFramebuffer(GL_FRAMEBUFFER, rend.gl.image_tx[1])
   # Re-render in the current mode
   update!(rend)
   render(rend)

   glBindTexture(GL_TEXTURE_2D, rend.gl.image_tx[1])
   glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pointer(rend.gl.data))

	glBindFramebuffer(GL_FRAMEBUFFER, 0)

   println("depth range: $(extrema(Float64.(rend.gl.data)))")

   image = colorview(RGBA, normedview(reshape(rend.gl.data, (4,rend.view.width,rend.view.height))))

	gl_check()
   save(file, transpose(image[:,end:-1:1]))
end

##################################

function simulate_depthcam(depth)
   tmp = imresize(depth, (400,400), method=BSpline(Constant()))
   imresize(tmp, (1600,1600), method=BSpline(Constant()))
end


function test_obj(name::String)
	obj = load(name)
	faces = IVector{3}.(SVector{3}.(obj.faces))
	vertices = SVector{3}.(obj.position)
end


end

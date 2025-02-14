module MeshRender

using StaticArrays, GLFW, ModernGL, LinearAlgebra, VisionGeometry,
Images, ImageTransformations, Interpolations

export Renderer, options!, viewing!

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
   diagm([scale,scale,scale,1]) * M * [rotation zeros(3,1); [0 0 0 1]]
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

"Initialize GL vector from concatenated array"
function gl_ptr(n::Int, gl_type::DataType=GLfloat)
   Ptr{Cvoid}(n*sizeof(gl_type))
end

@enum Render colour=1 depth=2 points=3 texture=4

mutable struct Renderer

	width::Int
   height::Int
	select::Tuple{Int,Int}

   window::GLFW.Window
	arc_press::GVector{3}
	arc_drag::Bool
   mode::Render
   opaque::Bool

	# Viewing geometry
	scale::Float64
	clip::Tuple{Float64,Float64}
	fov::Float64
	viewpoint::GVector{3}
	target::GVector{3}
	rotation::Matrix{Float64}
	rotation_pre::Matrix{Float64}
	
	# OpenGL data
   program::GLuint
   object_vao::GLuint
   mesh_vbo::GLuint
	points_vao::GLuint
   points_vbo::GLuint
	index_ranges::Vector{Tuple{Int32,Int32}}
	point_ranges::Vector{Tuple{Int32,Int32}}

   image_tx
   data::Array{GLuint,1}
	num_points::Int

   @doc """
       Renderer(window_size::Tuple{Int,Int}, V, F, N, C, P; 
                scale::Float64=1.0, clip=[0.1,30], 
                fov::Float64=60.0, viewpoint::AbstractVector=[0.0,0.0,5.0], target::AbstractVector=[0.0,0.0,0.0],
                visible=true)

   Construct a `Renderer`.
	"""
   function Renderer(window_size::Tuple{Int,Int}, 
		               V::Vector{GVectors{3}}, F::Vector{IVectors{3}},
							N::Vector{GVectors{3}}, C::Vector{GVectors{3}}, P::Vector{Vector{GVectors{3}}}; 
		               scale::Float64=1.0, clip::Tuple{Float64,Float64}=(0.1,30.0), fov::Float64=60.0,
							viewpoint::AbstractVector=[0.0,0.0,5.0], target::AbstractVector=[0.0,0.0,0.0],
							visible=true)

      rend = new(window_size[1], window_size[2], (1,1))

		GLFW.WindowHint(GLFW.SAMPLES, 4)
      GLFW.WindowHint(GLFW.OPENGL_DEBUG_CONTEXT, GL_TRUE)
      GLFW.WindowHint(GLFW.VISIBLE, visible)
      if !visible
         rend.data = Array{GLubyte,1}(undef, rend.width*rend.height*4)
      end
      rend.window = GLFW.CreateWindow(rend.width, rend.height, "Rendering mesh range $(rend.select)")
      GLFW.MakeContextCurrent(rend.window)
      glViewport(0,0,rend.width,rend.height)

		# Buffers for the merged data
      rend.object_vao = glGenVertexArray()
		rend.points_vao = glGenVertexArray()
		rend.mesh_vbo = glGenBuffer()
      rend.points_vbo = glGenBuffer()

		# Per-mesh (start,end) indices for the merged vertex data 
		K = cumsum(3*length.(F))
		rend.index_ranges = collect(zip([1;K.+1], K))
		K = cumsum(map(p -> sum(length.(p)), P))
		rend.point_ranges = collect(zip([1;K.+1], K))

      image_fb = Array{GLint,1}(undef,1)
      glGenFramebuffers(1,pointer(image_fb))
      glBindFramebuffer(GL_FRAMEBUFFER, image_fb[1])

      rend.image_tx = Array{GLuint,1}(undef,2)
      glGenTextures(1,pointer(rend.image_tx,1))
      glBindTexture(GL_TEXTURE_2D, rend.image_tx[1])
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rend.width, rend.height, 0, GL_RGB, GL_UNSIGNED_BYTE, gl_ptr(0))
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

      glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, rend.image_tx[1], 0)
      draw_bf = Array{GLenum,1}(undef,1)
      draw_bf[1] = GL_COLOR_ATTACHMENT0
      glDrawBuffers(1,pointer(draw_bf))
      status_fb = glCheckFramebufferStatus(GL_FRAMEBUFFER)
      glBindFramebuffer(GL_FRAMEBUFFER, image_fb[1])
      # println("Framebuffer ready: $(status_fb == GL_FRAMEBUFFER_COMPLETE)")
      # dbits = []
      # print("Depth bits: $(glGetIntegerv(GL_DEPTH_BITS, dbits))")

		# Viewing state
		rend.mode = colour
      rend.opaque = true
		rend.arc_press = @SVector[0.0, 0.0, 0.0]
		rend.arc_drag = false

		compile!(rend)
		buffers!(rend, V,F,N,C,P)
		viewing!(rend, scale, clip, fov, viewpoint, target)
      return rend
   end
end

""" Compile and link shaders.
"""
function compile!(rend::Renderer; vert_shader::String=vert_shader_default, 
	                               frag_shader::String=frag_shader_default)
   vsh = createShader(read(vert_shader,String), GL_VERTEX_SHADER)
   fsh = createShader(read(frag_shader,String), GL_FRAGMENT_SHADER)
   rend.program = createShaderProgram(vsh,fsh)
   glUseProgram(rend.program)
	options!(rend)
end

""" Set default options.
"""
function options!(rend::Renderer; background::AbstractVector=[211,215,207]/255, 
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
function viewing!(rend::Renderer, scale::Float64, clip::Tuple{Float64,Float64}, fov::Float64,
	                               viewpoint::Vector{Float64}, target::AbstractVector)
   rend.scale = scale
   rend.clip = clip
   rend.fov = fov
   rend.viewpoint = viewpoint
   rend.target = target
	rend.rotation = I(3)
	rend.rotation_pre = I(3)

   glUniform1f(glGetUniformLocation(rend.program,"near"), GLfloat(rend.clip[1]))
   glUniform1f(glGetUniformLocation(rend.program,"far"), GLfloat(rend.clip[2]))
   P = perspective(rend.clip, rend.fov, 1.0)
   glUniformMatrix4fv(glGetUniformLocation(rend.program,"projection"), 1, false, gl_vec(P))
end

""" Load data buffers for shaders.
"""
function buffers!(rend::Renderer, vertices, faces, normals, colours, points)

	# Merge data across all meshes
	V, N, C, P = vcat(vertices...), vcat(normals...), vcat(colours...), vcat(points...)

	# Add offsets to indices before merging across meshes
	F = map((f,n) -> f .+ [[n,n,n]], faces, [0; cumsum(length.(vertices))])
	# Merge indices across all meshes
	f = reduce(vcat,vcat(F...))

	# Treat each v/n/c as a 3x1 column, and form a block array from the K meshes: 
	#    V1 V2 ... VK
	#    N1 N2 ... VK
	#    C1 C2 ... VK
	# Then vec the entire array, columnwise.

	data = gl_vec(vcat(stack(V[f]), 
	                   stack(repeat(N,inner=3)),
							 stack(repeat(C,inner=3))))

	glBindVertexArray(rend.object_vao)
   glBindBuffer(GL_ARRAY_BUFFER, rend.mesh_vbo)
   glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
   stride = (3+3+3) * sizeof(GL_FLOAT)
   # vertices
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, gl_ptr(0))
   glEnableVertexAttribArray(0)
   # normals
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, gl_ptr(3))
   glEnableVertexAttribArray(1)
   # colours
   glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, gl_ptr(6))
   glEnableVertexAttribArray(2)

	### points
	rend.num_points = mapreduce(length,+,P)
	data_pts = gl_vec(reduce(vcat,reduce(vcat,P),init=[]))
	glBindVertexArray(rend.points_vao)
	glBindBuffer(GL_ARRAY_BUFFER, rend.points_vbo)
   glBufferData(GL_ARRAY_BUFFER, sizeof(data_pts), data_pts, GL_STATIC_DRAW)
   glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, gl_ptr(0))
   glEnableVertexAttribArray(3)
end

""" Render one frame.
"""
function render(rend::Renderer)
   if rend.mode == colour || rend.mode == depth
		k0 = first(rend.index_ranges[first(rend.select)])
		k1 = last(rend.index_ranges[last(rend.select)])
		glBindVertexArray(rend.object_vao)
      glDrawArrays(GL_TRIANGLES, k0-1, k1-k0+1)
	elseif rend.mode == points
		k0 = first(rend.point_ranges[first(rend.select)])
		k1 = last(rend.point_ranges[last(rend.select)])
		glBindVertexArray(rend.points_vao)
		glDrawArrays(GL_POINTS, k0-1, k1-k0+1)
	end
end

""" Update viewing geometry.
"""
function update!(rend::Renderer) 
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
   M = modelview(rend.viewpoint, rend.target, rotation=rend.rotation, scale=rend.scale)
   glUniformMatrix4fv(glGetUniformLocation(rend.program,"modelview"), 1, false, gl_vec(M))
end

""" Set interface callbacks and start the Renderer.
"""
function (rend::Renderer)()
	# Initalize as opaque
   glUniform1f(glGetUniformLocation(rend.program,"opacity"), GLfloat(1.0))

	# Key controls
   GLFW.SetKeyCallback(rend.window,
		(window::GLFW.Window, button::GLFW.Key, code::Int32, action::GLFW.Action, mods::Int32) ->
		begin
			if button == GLFW.KEY_ESCAPE
				GLFW.SetWindowShouldClose(rend.window,true)
			elseif button == GLFW.KEY_C && action == GLFW.PRESS
				rend.mode = colour
			elseif button == GLFW.KEY_D && action == GLFW.PRESS
				rend.mode = depth
			elseif button == GLFW.KEY_P && action == GLFW.PRESS
				rend.mode = points
			elseif button == GLFW.KEY_O && action == GLFW.PRESS
				rend.opaque = !rend.opaque
				glUniform1f(glGetUniformLocation(rend.program,"opacity"), GLfloat((1.0+rend.opaque)/2.0))
			elseif button == GLFW.KEY_LEFT && action == GLFW.PRESS
				# Merge and decrement the indices
				k = max(rend.select[1]-1, 1)
				rend.select = (k,k)
			elseif button == GLFW.KEY_RIGHT && action == GLFW.PRESS
				# Merge and increment the indices
				k = min(rend.select[1]+1, length(rend.index_ranges))
				rend.select = (k,k)
			elseif button == GLFW.KEY_UP && action == GLFW.PRESS
				# Add subsequent mesh
				rend.select = (rend.select[1], min(rend.select[2]+1, length(rend.index_ranges)))
			elseif button == GLFW.KEY_DOWN && action == GLFW.PRESS
				# Remove last mesh
				rend.select = (rend.select[1], max(rend.select[2]-1, rend.select[1]))
			end
			glUniform1i(glGetUniformLocation(rend.program,"render_mode"), GLint(rend.mode))
			GLFW.SetWindowTitle(rend.window, "Rendering mesh range $(rend.select)")
   	end)

	# Arcball button controls
	GLFW.SetMouseButtonCallback(rend.window,
		(window::GLFW.Window, button::GLFW.MouseButton, action::GLFW.Action, mods::Int32) ->
		begin
			if button == GLFW.MOUSE_BUTTON_LEFT
				if action == GLFW.PRESS
					# Store rotation state & start new arc
					rend.rotation_pre = rend.rotation
					rend.arc_press = arcball_vector(GVector{2}(GLFW.GetWindowSize(window)...), 
					                                GVector{2}(GLFW.GetCursorPos(window)...))
					rend.arc_drag = true
				elseif action == GLFW.RELEASE
					rend.arc_drag = false
				end
			end
		end)

	# Arcball drag controls
	GLFW.SetCursorPosCallback(rend.window,
		(window::GLFW.Window, x::Float64, y::Float64) ->
		begin
			if rend.arc_drag
				v = arcball_vector(GVector{2}(GLFW.GetWindowSize(window)...), GVector{2}(x,y))
				t = angle(rend.arc_press, v)
				n = cross(rend.arc_press, v)
				# Compose doubled differential rotation onto previous state
				rend.rotation = rotation(2.0*t,n) * rend.rotation_pre
			end
		end)
	
	# Scrollwheel zoom control
	GLFW.SetScrollCallback(rend.window,
		(window::GLFW.Window, x::Float64, y::Float64) ->
		begin
			rend.viewpoint = rend.viewpoint + [0.0, 0.0, y]
		end)

	# Window resizing/reshaping
  	GLFW.SetWindowSizeCallback(rend.window,
	   (window::GLFW.Window, w::Int32, h::Int32) ->
		begin
			P = perspective(rend.clip, rend.fov, w/h)
			glUniformMatrix4fv(glGetUniformLocation(rend.program,"projection"), 1, false, gl_vec(P))
			glViewport(0,0,w,h)
			rend.arc_drag = false
		end)

   glUniform1i(glGetUniformLocation(rend.program,"render_mode"), GLint(rend.mode))
   glBindFramebuffer(GL_FRAMEBUFFER,0)

   while !GLFW.WindowShouldClose(rend.window)
      update!(rend)
      render(rend)
      GLFW.SwapBuffers(rend.window)
      GLFW.PollEvents()
   end
   GLFW.DestroyWindow(rend.window)
end


function (rend::Renderer)(file::String, image_function=(depth)->depth)


   # Set first texture as target
   glBindFramebuffer(GL_FRAMEBUFFER, rend.image_tx[1])
   #rend.mode = texture
   update!(rend)
   render(rend)

   glBindTexture(GL_TEXTURE_2D, rend.image_tx[1])
   glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pointer(rend.data))
   println("depth range: $(extrema(Float64.(rend.data)))")

	tmp = reshape(rend.data, (4,rend.width,rend.height))

	println("$(size(rend.data)) : $(typeof(rend.data)) --> $(size(tmp)) : $(typeof(tmp))")

   img =  colorview(RGBA, normedview(N0f32,tmp))

	show(img[1:10,1:10])

	#println("$(size(img)) x $(typeof(img))")
   #println("depth range: $(extrema(Float64.(rend.data)))")

   save(file, img)
end

##################################

function simulate_depthcam(depth)
   tmp = imresize(depth, (400,400), method=BSpline(Constant()))
   imresize(tmp, (1600,1600), method=BSpline(Constant()))
end

end



#=


   # Set first texture as target
   glBindFramebuffer(GL_FRAMEBUFFER, rend.image_tx[1])
   # Re-render in the current mode
   update!(rend)
   render(rend)

   glBindTexture(GL_TEXTURE_2D, rend.image_tx[1])
   glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pointer(rend.data))
   println("depth range: $(extrema(Float64.(rend.data)))")

	tmp = reshape(rend.data, (rend.width,rend.height,4))

	println("$(size(tmp)) + $(typeof(tmp))")

   image = gl_image(rend.window) #colorview(RGBA{N0f8}, rend.data)
=#

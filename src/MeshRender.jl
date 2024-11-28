module MeshRender

using StaticArrays, GLFW, ModernGL, Images, ImageTransformations, Interpolations, LinearAlgebra, VisionGeometry

include(pkgdir(ModernGL, "test", "util.jl"))

vert_shader_default = pkgdir(@__MODULE__, "src", "MeshRenderVert.glsl")
frag_shader_default = pkgdir(@__MODULE__, "src", "MeshRenderFrag.glsl")

export Renderer, compile!, options!, buffers!, viewing!, execute!

""" Construct OpenGL camera matrix from [near,far] limits (unsigned),
    field of view (degrees), and aspect ratio.
"""
function perspective(limits, fov_v_deg, aspect=1)
   # Homogeneous perspective
   f = 1/tan(radians(fov_v_deg)/2)
   d = limits[2] - limits[1]
   cam = zeros(4,4)
   cam[1,1] = f / aspect
   cam[2,2] = f
   cam[3,3] = -sum(limits) / d
   cam[3,4] = -2*prod(limits) / d
   cam[4,3] = -1
   cam[4,4] = 0
   cam
end

""" Construct modelview matrix from coordinates of camera, 
    target, and up-vector.
"""
function viewing(cam, at, up)
   # Modelview matrix
   # Orientation
   v = normalize(at - cam)
   n = normalize(cross(v,up))
   u = normalize(cross(n,v))
   # Homogeneous form with translation
   [ n' -dot(n,cam);
     u' -dot(u,cam);
    -v'  dot(v,cam); 
     0 0 0 1 ]
end

function modelview(cam, at; up=[0; 1; 0], rotation=identity(3), scale=1)
   # Modelview matrix
   # Orientation
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

function gl_image(window)
   fb_size = GLFW.GetFramebufferSize(window)
   gl_data = gl_vec(Array{UInt8,1}(undef,prod(fb_size)*4), GLubyte)
   glPixelStorei(GL_PACK_ALIGNMENT, 4)
   glReadBuffer(GL_FRONT)
   glReadPixels(0, 0, GLint(fb_size[1]), GLint(fb_size[2]), 
                GL_RGBA, GL_UNSIGNED_BYTE, pointer(gl_data))
   colorview(RGBA, normedview(reshape(gl_data,(4,fb_size[1],fb_size[2]))))
   # imshow(img, axes=(2,1), flipy=true)
end

"Initialize GL vector from concatenated array"
function gl_vec(M, gl_type=GLfloat)
   Array{gl_type,1}(M[:])
end

"Initialize GL vector from concatenated array"
function gl_ptr(n, gl_type=GLfloat)
   Ptr{Cvoid}(n*sizeof(gl_type))
end

@enum Render colour=1 depth=2 points=3 texture=4

mutable struct Renderer

   width
   height
   window
   program
   object_vao
   mesh_vbo

	points_vao
   points_vbo
   image_tx

   data::Array{GLuint,1}
   num_vertices::Int
   num_faces::Int
	num_points::Int

   rotation
   scale
   clip
   fov
   viewpoint
   target

   mode::Render
   opaque::Bool

   function Renderer(w, h; visible=true, title="Renderer")

      rend = new(w,h)
      rend.rotation = identity(3)
      rend.scale = 1
      rend.mode = colour
      rend.opaque = true

      GLFW.WindowHint(GLFW.OPENGL_DEBUG_CONTEXT, GL_TRUE)
      GLFW.WindowHint(GLFW.VISIBLE, visible)
      if !visible
         rend.data = Array{GLubyte,1}(undef, rend.width*rend.height*4)
      end
      rend.window = GLFW.CreateWindow(rend.width, rend.height, title)
      GLFW.MakeContextCurrent(rend.window)
      glViewport(0,0,rend.width,rend.height)

      rend.object_vao = glGenVertexArray()
		rend.points_vao = glGenVertexArray()

		rend.mesh_vbo = glGenBuffer()
      rend.points_vbo = glGenBuffer()

      image_fb = Array{GLint,1}(undef,1)
      glGenFramebuffers(1,pointer(image_fb))
      glBindFramebuffer(GL_FRAMEBUFFER, image_fb[1])

      rend.image_tx = Array{GLint,1}(undef,2)
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
      println("Framebuffer ready: $(status_fb == GL_FRAMEBUFFER_COMPLETE)")
      dbits = []
      #print("Depth bits: $(glGetIntegerv(GL_DEPTH_BITS, dbits))")
      return(rend)
   end
end

function compile!(rend::Renderer, vert_shader::String=vert_shader_default, frag_shader::String=frag_shader_default)
   vsh = createShader(read(vert_shader,String), GL_VERTEX_SHADER)
   fsh = createShader(read(frag_shader,String), GL_FRAGMENT_SHADER)
   rend.program = createShaderProgram(vsh,fsh)
   glUseProgram(rend.program)
end

function options!(rend::Renderer)
   bg = [211,215,207]/255
   glClearColor(bg..., 1.0)
   glEnable(GL_BLEND)
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
   glEnable(GL_DEPTH_TEST)
   glDepthFunc(GL_LESS)
   glEnable(GL_PROGRAM_POINT_SIZE)
end

function viewing!(rend::Renderer; scale, clip, fov, viewpoint, target)
   rend.scale = scale
   rend.clip = clip
   rend.fov = fov
   rend.viewpoint = viewpoint
   rend.target = target
   glUniform1f(glGetUniformLocation(rend.program,"near"), GLfloat(rend.clip[1]))
   glUniform1f(glGetUniformLocation(rend.program,"far"), GLfloat(rend.clip[2]))
   P = perspective(rend.clip, rend.fov, 1)
   #display(P)
   glUniformMatrix4fv(glGetUniformLocation(rend.program,"projection"), 1, false, gl_vec(P[:]))
end

function buffers!(rend::Renderer, vertices, faces, normals, colours, points)

   rend.num_vertices = length(vertices)
   rend.num_faces = length(faces)
   #data = gl_vec(vcat(vertices[:,faces[:]], 
   #              repeat(normals,inner=[1,3]), 
   #              repeat(colours,inner=[1,3]))[:])

	data = gl_vec(vcat(stack(vertices[reduce(vcat,faces)]),
	            stack(repeat(normals, inner=3)),  
					stack(repeat(colours, inner=3))))


   glBindVertexArray(rend.object_vao)
   glBindBuffer(GL_ARRAY_BUFFER, rend.mesh_vbo)
   glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
   stride = (3 + 3 + 3) * sizeof(GL_FLOAT)
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
	rend.num_points = mapreduce(length,+,points)
	println("loading $(rend.num_points) pts")
	data_pts = gl_vec(reduce(vcat,reduce(vcat,points)))

	glBindVertexArray(rend.points_vao)
	glBindBuffer(GL_ARRAY_BUFFER, rend.points_vbo)
   glBufferData(GL_ARRAY_BUFFER, sizeof(data_pts), data_pts, GL_STATIC_DRAW)
   glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, gl_ptr(0))
   glEnableVertexAttribArray(3)

end

function render(rend::Renderer)
   if rend.mode == colour || rend.mode == depth
      glBindVertexArray(rend.object_vao)
      glDrawArrays(GL_TRIANGLES, 0, 3*rend.num_faces)
   end
	###
	if rend.mode == points
		glBindVertexArray(rend.points_vao)
		glDrawArrays(GL_POINTS, 0, 1*rend.num_points)
	end
end

theta = 0.0
function update!(rend::Renderer) 
   rend.rotation = rotation(theta, @SVector[1.0,1.0,1.0])
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
   M = modelview(rend.viewpoint, rend.target, rotation=rend.rotation, scale=rend.scale)
   glUniformMatrix4fv(glGetUniformLocation(rend.program,"modelview"), 1, false, gl_vec(M[:]))
   global theta += 0.0025 * 1
end

function execute!(rend::Renderer)

   GLFW.SetKeyCallback(rend.window, (window, button, code, action, modifier) ->
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
			println("$((1.0+rend.opaque)/2)")
         glUniform1f(glGetUniformLocation(rend.program,"opacity"), GLfloat((1.0+rend.opaque)/2))
      end
      glUniform1i(glGetUniformLocation(rend.program,"render_mode"), GLint(rend.mode))
   end)
   glUniform1i(glGetUniformLocation(rend.program,"render_mode"), GLint(rend.mode))
   #glUniform1f(glGetUniformLocation(rend.program,"opacity"), GLfloat(1.0))
   glBindFramebuffer(GL_FRAMEBUFFER,0)

   while !GLFW.WindowShouldClose(rend.window)
      update!(rend)
      render(rend)
      GLFW.SwapBuffers(rend.window)
      GLFW.PollEvents()
   end
   GLFW.DestroyWindow(rend.window)
end

function execute!(rend::Renderer, file::String, image_function=(depth)->depth)

   glUniform1i(glGetUniformLocation(rend.program,"render_mode"), GLint(rend.mode))
   glUniform1f(glGetUniformLocation(rend.program,"opacity"), GLfloat(1.0))

   # Set first texture as target
   glBindFramebuffer(GL_FRAMEBUFFER, rend.image_tx[1])
   # Re-render in the current mode
   update!(rend)
   render(rend)

   glBindTexture(GL_TEXTURE_2D, rend.image_tx[1])
   glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pointer(rend.data))
   println("depth range: $(extrema(Float64.(rend.data)))")
   image = colorview(RGBA, normedview(reshape(rend.data, (4,rend.width,rend.height))))
   save(file, image_function(image))
end

##################################

function simulate_kinect(depth)
   tmp = imresize(depth, (400,400), method=BSpline(Constant()))
   imresize(tmp, (1600,1600), method=BSpline(Constant()))
end

end

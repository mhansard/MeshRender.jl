#version 400 
layout(location=0) in vec3 vertex;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 texcoord;


//layout(location=2) in vec3 colour;
//layout(location=3) in vec3 point;
//layout(location=4) in vec2 texcoord;

uniform mat4 modelview, projection;
uniform int render_mode;
//uniform float opacity;
uniform float near, far;

out highp vec3 s, n;
out mediump vec4 c;
out highp vec2 map;

// physical depth
out float z;

vec3 light = vec3(5.0, 5.0, 0.0);

void main()
{
    gl_PointSize = 5.0;
    if(render_mode == 1 || render_mode == 2) {
        gl_Position = projection * modelview * vec4(vertex,1.0);
        z = (modelview * vec4(vertex,1.0)).z;
        s = normalize(light - vertex.xyz);
        n = normalize((modelview * vec4(normal,0.0)).xyz);
        c = vec4(0.0);///= vec4(colour.rgb, 1.0);
    }
    else if(render_mode == 3) {
        gl_Position = vec4(0.0);///projection * modelview * vec4(point,1.0);
        s = vec3(0.0);
        n = vec3(0.0);
        c = vec4(0.0);
    }
    else if(render_mode == 4) {
		  gl_Position = projection * modelview * vec4(vertex,1.0);
		  map = texcoord;
        //gl_Position = vec4(corner.x, corner.y, 0.0, 1.0); 
        //map = (vec2(1.0,1.0) + corner) / 2.0;    
    }
}

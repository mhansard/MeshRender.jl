#version 400 

precision highp float;
uniform int render_mode;
uniform float near, far;

uniform sampler2D image;
uniform float opacity;

in highp vec3 s, n;
in mediump vec4 c;
in vec2 map;
in float z;

layout(location = 0) out mediump vec4 colour;

float scene_depth(float frag_z)
{
   float ndc_z = 2.0*frag_z - 1.0;
   return (2.0*near*far) / (far + near - ndc_z*(far-near));
}

void main()
{
   if(render_mode == 1) {
      mediump float lambert = max(dot(normalize(s), normalize(n)), 0.0);
      colour = vec4(c.rgb * (0.5 + 0.5*lambert), opacity);
   }
   else if(render_mode == 2) {
      // Scaled depth: near <= abs(z) <= far 
      float c  = (abs(z)-far) / (near-far);
      if(abs(z) < near)
         colour = vec4(0.0, 0.0, 1.0, 1.0);
      else if(abs(z) > far)
         colour = vec4(1.0, 0.0, 0.0, 1.0);
      else
         colour = vec4(vec3(c), 1.0);
   }
   else if(render_mode == 3) {
      colour = vec4(vec3(0.2), 1.0);
   }
}

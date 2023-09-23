#version 450 core
//layout (location = 0) in vec2 aPos;
//layout (location = 1) in vec2 aTexCoord;

layout (location = 0) out vec2 ourTexCoord;

void main()
{
//    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
//    ourTexCoord = aTexCoord;

    float corner = 0.3;

    if(gl_VertexIndex == 0) gl_Position = vec4(-1, -1, 1, 1);
    if(gl_VertexIndex == 1) gl_Position = vec4(-1,  -corner, 1, 1);
    if(gl_VertexIndex == 2) gl_Position = vec4( -corner, -1, 1, 1);
    if(gl_VertexIndex == 3) gl_Position = vec4( -corner, -corner, 1, 1);
    if(gl_VertexIndex == 4) gl_Position = vec4( -corner, -1, 1, 1);
    if(gl_VertexIndex == 5) gl_Position = vec4( -1, -corner, 1, 1);
    vec2 coord = (gl_Position.xy + vec2(1)) / vec2(0.7); // * vec2(0.5, -0.5) + vec2(0.5);
    ourTexCoord = vec2(coord.x, 1 - coord.y);
}
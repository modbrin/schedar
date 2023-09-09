#version 450 core
//layout (location = 0) in vec2 aPos;
//layout (location = 1) in vec2 aTexCoord;

layout (location = 0) out vec2 ourTexCoord;

void main()
{
//    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
//    ourTexCoord = aTexCoord;

    if(gl_VertexIndex == 0) gl_Position = vec4(-1, -1, 1, 1);
    if(gl_VertexIndex == 1) gl_Position = vec4(-1,  3, 1, 1);
    if(gl_VertexIndex == 2) gl_Position = vec4( 3, -1, 1, 1);
    ourTexCoord = gl_Position.xy * vec2(0.5, -0.5) + vec2(0.5);
}
#version 450
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
//layout (location = 3) in vec3 aTangent;
//layout (location = 4) in vec3 aBitangent;

layout (set = 0, binding = 0) uniform Uniforms {
    mat4 model;
    mat4 mvp;
    mat4 normal;
    vec3 viewPos;
} uniforms;

layout (location = 0) out vec3 ourViewPos;
layout (location = 1) out vec3 ourFragPos;
layout (location = 2) out vec3 ourNormal;
layout (location = 3) out vec2 ourTexCoord;

void main()
{
    gl_Position = uniforms.mvp * vec4(aPos, 1.0);
    ourTexCoord = vec2(aTexCoord.s, aTexCoord.t);
    ourFragPos = vec3(uniforms.model * vec4(aPos, 1.0));
    ourNormal = vec3(uniforms.normal * vec4(aNormal, 0.0)); // TODO: supply proper 3x3 normal matrix
    ourViewPos = uniforms.viewPos;
}
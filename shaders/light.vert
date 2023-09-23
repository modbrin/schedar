#version 450
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
//layout (location = 3) in vec3 aTangent;
//layout (location = 4) in vec3 aBitangent;

layout (set = 0, binding = 0) uniform Uniforms {
    mat4 viewProject;
    vec3 viewPos;
    mat4 lightSpaceViewProject;
} uniforms;

layout (location = 0) out vec3 ourViewPos;
layout (location = 1) out vec3 ourFragPos;
layout (location = 2) out vec3 ourNormal;
layout (location = 3) out vec2 ourTexCoord;
layout (location = 4) out vec4 ourFragPosLightSpace;

layout (push_constant) uniform PushConstants {
    mat4 model;
    mat4 normal;
} pConsts;

void main()
{
    ourViewPos = uniforms.viewPos;
    ourFragPos = vec3(pConsts.model * vec4(aPos, 1.0));
    ourNormal = vec3(pConsts.normal * vec4(aNormal, 0.0)); // TODO: supply proper 3x3 normal matrix
    ourTexCoord = vec2(aTexCoord.s, aTexCoord.t);
    ourFragPosLightSpace = uniforms.lightSpaceViewProject * vec4(ourFragPos, 1.0);
    gl_Position = uniforms.viewProject * vec4(ourFragPos, 1.0);
}
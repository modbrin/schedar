#version 450 core
layout (location = 0) in vec3 aPos;

layout (push_constant) uniform PushConstants {
    mat4 lightSpaceMatrix;
    mat4 model;
} pConsts;

void main()
{
    gl_Position = pConsts.lightSpaceMatrix * pConsts.model * vec4(aPos, 1.0);
}
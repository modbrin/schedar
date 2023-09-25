#version 450 core
layout (location = 0) out vec4 FragColor;

layout (location = 0) in vec2 ourTexCoord;

layout(set = 0, binding = 0) uniform texture2D depthTexture;
layout(set = 0, binding = 1) uniform sampler textureSampler;

void main()
{
    float depth = texture(sampler2D(depthTexture, textureSampler), ourTexCoord).r;

//    if (depth > 0.999) {
//        FragColor = vec4(1.0, 0.0, 0.0, 1.0);
//    } else {
//        FragColor = vec4(0.0, 1.0, 0.0, 1.0);
//    }
    FragColor = vec4(vec3((depth - 0.99) * 100), 1.0);
}
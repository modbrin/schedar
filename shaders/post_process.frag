#version 450 core
layout (location = 0) out vec4 FragColor;

layout (location = 0) in vec2 ourTexCoord;

//uniform sampler2D screenTexture;

layout(set = 0, binding = 0) uniform texture2D screenTexture;
layout(set = 0, binding = 1) uniform sampler screenTextureSampler;

const float offset = 1.0 / 200.0;


float sharpenKernel[9] = float[](
    -1, -1, -1,
    -1,  9, -1,
    -1, -1, -1
);

float blurKernel[9] = float[](
    1.0 / 16, 2.0 / 16, 1.0 / 16,
    2.0 / 16, 4.0 / 16, 2.0 / 16,
    1.0 / 16, 2.0 / 16, 1.0 / 16
);

float edgeDetectionKernel[9] = float[](
    1,  1,  1,
    1, -8,  1,
    1,  1,  1
);

void main()
{
    vec4 color = texture(sampler2D(screenTexture, screenTextureSampler), ourTexCoord);

    // gamma correction
//    float gamma = 2.2;
//    color.rgb = pow(color.rgb, vec3(1.0/gamma));

    // no effects
    FragColor = color;

    // inverse
//    FragColor = vec4(color, 1);

    // grayscale
//       FragColor = color;
//       float average = 0.2126 * FragColor.r + 0.7152 * FragColor.g + 0.0722 * FragColor.b;
//       FragColor = vec4(average, average, average, 1.0);

    // kernel effects
//    vec2 offsets[9] = vec2[](
//        vec2(-offset,  offset), // top-left
//        vec2( 0.0f,    offset), // top-center
//        vec2( offset,  offset), // top-right
//        vec2(-offset,  0.0f),   // center-left
//        vec2( 0.0f,    0.0f),   // center-center
//        vec2( offset,  0.0f),   // center-right
//        vec2(-offset, -offset), // bottom-left
//        vec2( 0.0f,   -offset), // bottom-center
//        vec2( offset, -offset)  // bottom-right
//    );
//
//    vec3 sampleTex[9];
//    for(int i = 0; i < 9; i++)
//    {
//        sampleTex[i] = color.rgb;
//    }
//    vec3 col = vec3(0.0);
//    for(int i = 0; i < 9; i++) {
//        col += sampleTex[i] * sharpenKernel[i];
//    }
//
//    FragColor = vec4(col.rgb, 1.0);
}
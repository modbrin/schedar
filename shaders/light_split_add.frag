#version 450 core
layout (location = 0) out vec4 FragColor;

layout (location = 0) in vec3 ourViewPos;
layout (location = 1) in vec3 ourFragPos;
layout (location = 2) in vec3 ourNormal;
layout (location = 3) in vec2 ourTexCoord;

#define NR_POINT_LIGHTS 16
#define NR_SPOT_LIGHTS 16

struct SpotLight {
    vec3 position;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float cutoff;
    float outerCutoff;
    float constant;
    float linear;
    float quadratic;
};

struct PointLight {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};

layout(set = 1, binding = 0) uniform LightsUniform {
    PointLight pointLights[NR_POINT_LIGHTS];
    uint pointCount;
} lights;

//layout(set = 1, binding = 1) uniform SpotLightsUniform {
//    SpotLight spotLights[NR_SPOT_LIGHTS];
//    uint spotCount;
//} spotLights;


layout(set = 2, binding = 0) uniform MaterialParams {
    float shininess;
} materialParams;

// diffuse
layout(set = 2, binding = 1) uniform texture2D texture_diffuse1;
layout(set = 2, binding = 2) uniform sampler sampler_diffuse1;
// specular
layout(set = 2, binding = 5) uniform texture2D texture_specular1;
layout(set = 2, binding = 6) uniform sampler sampler_specular1;
// normal
layout(set = 2, binding = 9) uniform texture2D texture_normal1;
layout(set = 2, binding = 10) uniform sampler sampler_normal1;
//emissive
layout(set = 2, binding = 13) uniform texture2D texture_emissive1;
layout(set = 2, binding = 14) uniform sampler sampler_emissive1;


vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - fragPos);
    vec3 halfDir = normalize(lightDir + viewDir);
    float diff = max(dot(normal, lightDir), 0.0);
//    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(normal, halfDir), 0.0), materialParams.shininess);

    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

    vec3 diffuseTex = vec3(texture(sampler2D(texture_diffuse1, sampler_diffuse1), ourTexCoord));
    vec3 ambient  = light.ambient * diffuseTex;
    vec3 diffuse  = light.diffuse * diff * diffuseTex;
    vec3 specular = light.specular * spec * vec3(texture(sampler2D(texture_specular1, sampler_specular1), ourTexCoord));
    return (ambient + diffuse + specular) * attenuation;

}

//vec3 debugBinaryVec(vec3 v) {
//    if (gl_FragCoord.x < 250.0) {
//        if (v.r == 1.0) {
//            return vec3(1.0, 1.0, 1.0);
//        } else {
//            return vec3(0.0, 0.0, 0.0);
//        }
//    }
//    if (gl_FragCoord.x >= 250.0 && (gl_FragCoord.x < 500.0)) {
//        if (v.g == 1.0) {
//            return vec3(1.0, 1.0, 1.0);
//        } else {
//            return vec3(0.0, 0.0, 0.0);
//        }
//    }
//    if (gl_FragCoord.x >= 500.0) {
//        if (v.b == 1.0) {
//            return vec3(1.0, 1.0, 1.0);
//        } else {
//            return vec3(0.0, 0.0, 0.0);
//        }
//    }
//}

//vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
//{
//    vec3 lightDir = normalize(light.position - fragPos);
//    float diff = max(dot(normal, lightDir), 0.0);
//    vec3 reflectDir = reflect(-lightDir, normal);
//    float spec = pow(max(dot(viewDir, reflectDir), 0.0), materialParams.shininess);
//
//    float distance = length(light.position - fragPos);
//    float attenuation = 1.0 / (light.constant + light.linear * distance +
//    light.quadratic * (distance * distance));
//
//    vec3 ambient  = light.ambient  * vec3(texture(texture_diffuse1, ourTexCoord));
//    vec3 diffuse  = light.diffuse  * diff * vec3(texture(texture_diffuse1, ourTexCoord));
//    vec3 specular = light.specular * spec * vec3(texture(texture_specular1, ourTexCoord));
//
//    float theta = dot(lightDir, normalize(-light.direction));
//    float epsilon = light.cutoff - light.outerCutoff;
//    float intensity = clamp((theta - light.outerCutoff) / epsilon, 0.0, 1.0);
//    return (ambient + diffuse * intensity + specular * intensity) * attenuation;
//}

//float near = 0.1;
//float far  = 100.0;
//
//float LinearizeDepth(float depth)
//{
//    float z = depth * 2.0 - 1.0; // back to NDC
//    return (2.0 * near * far) / (far + near - z * (far - near));
//}

void main()
{
    vec3 norm = normalize(ourNormal);
    vec3 viewDir = normalize(ourViewPos - ourFragPos);

    vec3 result = vec3(0);

    for(int i = 0; i < lights.pointCount; i++) {
        result += 0.7 * CalcPointLight(lights.pointLights[i], norm, ourFragPos, viewDir);
    }

//    result += CalcSpotLight(spotLight, norm, ourFragPos, viewDir);

//    float depth = LinearizeDepth(gl_FragCoord.z) / far;

    float diffuseAlpha = texture(sampler2D(texture_diffuse1, sampler_diffuse1), ourTexCoord).a; // TODO: is additive alpha needed at all?
    if (diffuseAlpha < 0.1)
        discard;

//    vec3 pureDiffuse = vec3(texture(material.texture_diffuse1, ourTexCoord)) + vec3(texture(material.texture_diffuse2, ourTexCoord));

//    FragColor = vec4(mix(vec3(depth), pureDiffuse, 0.5), diffuseAlpha);
    FragColor = vec4(vec3(result), diffuseAlpha);
//    FragColor = vec4(vec3(depth), 1.0);
}



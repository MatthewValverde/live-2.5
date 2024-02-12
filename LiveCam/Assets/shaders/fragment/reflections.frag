#version 450
layout (location = 0) out vec4 FragColor;

in vec3 Position;
in vec3 Normal;
in vec2 TexCoord;

uniform sampler2D Texture;
uniform samplerCube CubeMap;
uniform vec4 cameraPos;

void main()
{
    vec3 norm = normalize(Normal);
    vec3 I = normalize(Position - vec3(cameraPos));
    vec3 R = reflect(I, norm);
    FragColor = vec4(texture(CubeMap, R).rgb, 1.0);
}

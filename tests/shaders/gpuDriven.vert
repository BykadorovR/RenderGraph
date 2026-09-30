#version 460

layout(set = 0, binding = 0, std430) readonly buffer Positions {
  vec4 values[];
} positions;

void main() {
  gl_Position = positions.values[gl_VertexIndex];
}

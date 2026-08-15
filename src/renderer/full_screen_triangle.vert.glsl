
// Oversized triangle covering the whole screen; no vertex buffer needed.
void main() {
    switch (gl_VertexIndex) {
    case 0: gl_Position = vec4(-1, -1, 0, 1); break;
    case 1: gl_Position = vec4(-1, +4, 0, 1); break;
    case 2: gl_Position = vec4(+4, -1, 0, 1); break;
    }
}

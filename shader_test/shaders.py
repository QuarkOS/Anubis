"""
GLSL shader sources for the avatar overlay.
"""

# ── Fullscreen-quad vertex shader ─────────────────────────────

VERTEX_SHADER = """\
#version 330
in vec2 in_position;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

# ── Fragment wrapper (injects Shadertoy-style uniforms) ───────

FRAGMENT_WRAPPER = """\
#version 330
out vec4 fragColor;
uniform vec2  iResolution;
uniform float iTime;
uniform vec2  iMouse;

{shader_body}

void main() {{
    mainImage(fragColor, gl_FragCoord.xy);
}}
"""

# ── Volumetric cloud sphere (based on Shadertoy lss3zr) ──────

CLOUD_SPHERE = """\
mat3 m = mat3(0.00, 0.80, 0.60,
             -0.80, 0.36,-0.48,
             -0.60,-0.48, 0.64);

float hash(float n) {
    return fract(sin(n) * 43758.5453);
}

float noise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    return mix(
        mix(mix(hash(n),       hash(n+1.0),   f.x),
            mix(hash(n+57.0),  hash(n+58.0),  f.x), f.y),
        mix(mix(hash(n+113.0), hash(n+114.0), f.x),
            mix(hash(n+170.0), hash(n+171.0), f.x), f.y),
        f.z
    );
}

float fbm(vec3 p) {
    float f  = 0.5000 * noise(p); p = m * p * 2.02;
          f += 0.2500 * noise(p); p = m * p * 2.03;
          f += 0.1250 * noise(p);
    return f;
}

float scene(vec3 pos) {
    return 0.1 - length(pos) * 0.05 + fbm(pos * 0.3);
}

mat3 camera(vec3 ro, vec3 ta) {
    vec3 cw = normalize(ta - ro);
    vec3 cu = cross(cw, vec3(0.0, 1.0, 0.0));
    vec3 cv = cross(cu, cw);
    return mat3(cu, cv, cw);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);

    // ── Sphere mask ──
    float dist = length(uv);
    if (dist > 1.0) { fragColor = vec4(0.0); return; }
    float mask = smoothstep(1.0, 0.97, dist);

    // ── Camera ──
    vec2  mo = vec2(iTime * 0.1, cos(iTime * 0.25) * 0.8);
    vec3  ta = vec3(0.0, 1.0, 0.0);
    vec3  ro = 25.0 * normalize(vec3(
                   cos(2.75 - 3.0 * mo.x),
                   0.7 - (mo.y - 1.0),
                   sin(2.75 - 3.0 * mo.x)));
    mat3  c  = camera(ro, ta);
    vec3 dir = c * normalize(vec3(uv, 1.3));

    // ── Raymarching ──
    const int   N     = 64;
    const float zstep = 40.0 / float(N);
    vec3  p     = ro;
    float T     = 1.0;
    vec4  color = vec4(0.0);

    for (int i = 0; i < N; i++) {
        float density = scene(p);
        if (density > 0.0) {
            float tmp = density / float(N);
            T *= 1.0 - tmp * 100.0;
            if (T <= 0.01) break;
            color += vec4(1.0) * (50.0 * tmp * T);
        }
        p += dir * zstep;
    }

    // ── Background gradient (purple → light blue) ──
    color.rgb += mix(vec3(0.3, 0.1, 0.8),
                     vec3(0.7, 0.7, 1.0),
                     1.0 - (uv.y + 1.0) * 0.5);

    fragColor = vec4(color.rgb, mask);
}
"""

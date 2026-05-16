shader_type canvas_item;

// Volumetric Cloud Raymarching cloned from https://www.shadertoy.com/view/lss3zr

#define USE_LIGHT 0

const mat3 m = mat3(
    vec3(0.00, 0.80, 0.60),
    vec3(-0.80, 0.36, -0.48),
    vec3(-0.60, -0.48, 0.64)
);

float hash(float n) {
    return fract(sin(n) * 43758.5453);
}

float noise(in vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    float res = mix(mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
                        mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
                    mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                        mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
    return res;
}

float fbm(vec3 p) {
    float f;
    f = 0.5000 * noise(p); p = m * p * 2.02;
    f += 0.2500 * noise(p); p = m * p * 2.03;
    f += 0.1250 * noise(p);
    return f;
}

float scene(in vec3 pos) {
    return 0.1 - length(pos) * 0.05 + fbm(pos * 0.3);
}

mat3 camera(vec3 ro, vec3 ta) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(0.0, 1.0, 0.0);
    vec3 cu = cross(cw, cp);
    vec3 cv = cross(cu, cw);
    return mat3(cu, cv, cw);
}

void fragment() {
    vec2 iResolution = 1.0 / SCREEN_PIXEL_SIZE;
    vec2 uv = (FRAGCOORD.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);
    
    vec2 mo = vec2(TIME * 0.1, cos(TIME * 0.25) * 3.0);
    
    float camDist = 25.0;
    vec3 ta = vec3(0.0, 1.0, 0.0);
    vec3 ro = camDist * normalize(vec3(cos(2.75 - 3.0 * mo.x), 0.7 - 1.0 * (mo.y - 1.0), sin(2.75 - 3.0 * mo.x)));
    
    float targetDepth = 1.3;
    mat3 c = camera(ro, ta);
    vec3 dir = c * normalize(vec3(uv, targetDepth));
    
    const int sampleCount = 64;
    float zMax = 40.0;
    float zstep = zMax / float(sampleCount);
    
    vec3 p = ro;
    float T = 1.0;
    float absorption = 100.0;
    vec4 color = vec4(0.0);
    
    for (int i = 0; i < sampleCount; i++) {
        float density = scene(p);
        if (density > 0.0) {
            float tmp = density / float(sampleCount);
            T *= 1.0 - (tmp * absorption);
            if (T <= 0.01) break;
            
            float opaity = 50.0;
            float k = opaity * tmp * T;
            vec4 cloudColor = vec4(1.0);
            vec4 col1 = cloudColor * k;
            color += col1;
        }
        p += dir * zstep;
    }
    
    vec3 bg = mix(vec3(0.3, 0.1, 0.8), vec3(0.7, 0.7, 1.0), 1.0 - (uv.y + 1.0) * 0.5);
    color.rgb += bg;
    
    COLOR = color;
}

"""
Shadertoy Shader Performance Tester
====================================
Press SPACE or click to toggle between shaders.
Press ESC to exit.
"""

import moderngl
import pyglet
from pyglet.gl import *
import time as _time

# ── Shader Sources ──────────────────────────────────────────────

SHADER_NAMES = [
    "Shader 1: Ring Noise (WslGWl)",
    "Shader 2: Fractal Pyramid (tsXBzS)",
    "Shader 3: Protean Clouds (3l23Rh)",
    "Shader 4: Volumetric Clouds (lss3zr)",
]

shader1_src = """
vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(.1031,.11369,.13787));
    p3 += dot(p3, p3.yxz+19.19);
    return -1.0 + 2.0 * fract(vec3(p3.x+p3.y, p3.x+p3.z, p3.y+p3.z)*p3.zyx);
}
float snoise3(vec3 p) {
    const float K1 = 0.333333333;
    const float K2 = 0.166666667;
    vec3 i = floor(p + (p.x + p.y + p.z) * K1);
    vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);
    vec3 e = step(vec3(0.0), d0 - d0.yzx);
    vec3 i1 = e * (1.0 - e.zxy);
    vec3 i2 = 1.0 - e.zxy * (1.0 - e);
    vec3 d1 = d0 - (i1 - K2);
    vec3 d2 = d0 - (i2 - K1);
    vec3 d3 = d0 - 0.5;
    vec4 h = max(0.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
    vec4 n = h * h * h * h * vec4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0)));
    return dot(vec4(31.316), n);
}
vec4 extractAlpha(vec3 colorIn) {
    vec4 colorOut;
    float maxValue = min(max(max(colorIn.r, colorIn.g), colorIn.b), 1.0);
    if (maxValue > 1e-5) {
        colorOut.rgb = colorIn.rgb * (1.0 / maxValue);
        colorOut.a = maxValue;
    } else {
        colorOut = vec4(0.0);
    }
    return colorOut;
}
const vec3 color1 = vec3(0.611765, 0.262745, 0.996078);
const vec3 color2 = vec3(0.298039, 0.760784, 0.913725);
const vec3 color3 = vec3(0.062745, 0.078431, 0.600000);
const float innerRadius = 0.6;
const float noiseScale = 0.65;
float light1(float intensity, float attenuation, float dist) {
    return intensity / (1.0 + dist * attenuation);
}
float light2(float intensity, float attenuation, float dist) {
    return intensity / (1.0 + dist * dist * attenuation);
}
void draw_ring(out vec4 _FragColor, in vec2 vUv) {
    vec2 uv = vUv;
    float ang = atan(uv.y, uv.x);
    float len = length(uv);
    float n0 = snoise3(vec3(uv * noiseScale, iTime * 0.5)) * 0.5 + 0.5;
    float r0 = mix(mix(innerRadius, 1.0, 0.4), mix(innerRadius, 1.0, 0.6), n0);
    float d0 = distance(uv, r0 / len * uv);
    float v0 = light1(1.0, 10.0, d0);
    v0 *= smoothstep(r0 * 1.05, r0, len);
    float cl = cos(ang + iTime * 2.0) * 0.5 + 0.5;
    float a = iTime * -1.0;
    vec2 pos = vec2(cos(a), sin(a)) * r0;
    float d = distance(uv, pos);
    float v1 = light2(1.5, 5.0, d);
    v1 *= light1(1.0, 50.0, d0);
    float v2 = smoothstep(1.0, mix(innerRadius, 1.0, n0 * 0.5), len);
    float v3 = smoothstep(innerRadius, mix(innerRadius, 1.0, 0.5), len);
    vec3 col = mix(color1, color2, cl);
    col = mix(color3, col, v0);
    col = (col + v1) * v2 * v3;
    col = clamp(col, 0.0, 1.0);
    _FragColor = extractAlpha(col);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    vec4 col;
    draw_ring(col, uv);
    fragColor.rgb = mix(vec3(0.0), col.rgb, col.a);
    fragColor.a = 1.0;
}
"""

shader2_src = """
vec3 palette(float d) {
    return mix(vec3(0.2,0.7,0.9), vec3(1.,0.,1.), d);
}
vec2 rotate(vec2 p, float a) {
    float c = cos(a), s = sin(a);
    return p * mat2(c,s,-s,c);
}
float map(vec3 p) {
    for (int i = 0; i < 8; ++i) {
        float t = iTime * 0.2;
        p.xz = rotate(p.xz, t);
        p.xy = rotate(p.xy, t * 1.89);
        p.xz = abs(p.xz);
        p.xz -= .5;
    }
    return dot(sign(p), p) / 5.;
}
vec4 rm(vec3 ro, vec3 rd) {
    float t = 0.;
    vec3 col = vec3(0.);
    float d;
    for (float i = 0.; i < 64.; i++) {
        vec3 p = ro + rd * t;
        d = map(p) * .5;
        if (d < 0.02) break;
        if (d > 100.) break;
        col += palette(length(p) * .1) / (400. * d);
        t += d;
    }
    return vec4(col, 1. / (d * 100.));
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - iResolution.xy / 2.) / iResolution.x;
    vec3 ro = vec3(0., 0., -50.);
    ro.xz = rotate(ro.xz, iTime);
    vec3 cf = normalize(-ro);
    vec3 cs = normalize(cross(cf, vec3(0., 1., 0.)));
    vec3 cu = normalize(cross(cf, cs));
    vec3 uuv = ro + cf * 3. + uv.x * cs + uv.y * cu;
    vec3 rd = normalize(uuv - ro);
    vec4 col = rm(ro, rd);
    fragColor = col;
    fragColor.a = 1.0;
}
"""

shader3_src = """
mat2 rot(in float a){float c = cos(a), s = sin(a); return mat2(c,s,-s,c);}
const mat3 m3 = mat3(0.33338, 0.56034, -0.71817, -0.87887, 0.32651, -0.15323, 0.15162, 0.69596, 0.61339)*1.93;
float mag2(vec2 p){return dot(p,p);}
float linstep(in float mn, in float mx, in float x){ return clamp((x - mn)/(mx - mn), 0., 1.); }
float prm1 = 0.;
vec2 bsMo = vec2(0);
vec2 disp(float t){ return vec2(sin(t*0.22), cos(t*0.175))*2.; }
vec2 map(vec3 p) {
    vec3 p2 = p;
    p2.xy -= disp(p.z).xy;
    p.xy *= rot(sin(p.z+iTime)*(0.1 + prm1*0.05) + iTime*0.09);
    float cl = mag2(p2.xy);
    float d = 0.;
    p *= .61;
    float z = 1.;
    float trk = 1.;
    float dspAmp = 0.1 + prm1*0.2;
    for(int i = 0; i < 5; i++) {
        p += sin(p.zxy*0.75*trk + iTime*trk*.8)*dspAmp;
        d -= abs(dot(cos(p), sin(p.yzx))*z);
        z *= 0.57;
        trk *= 1.4;
        p = p*m3;
    }
    d = abs(d + prm1*3.)+ prm1*.3 - 2.5 + bsMo.y;
    return vec2(d + cl*.2 + 0.25, cl);
}
vec4 render(in vec3 ro, in vec3 rd, float time) {
    vec4 rez = vec4(0);
    float t = 1.5;
    float fogT = 0.;
    for(int i=0; i<130; i++) {
        if(rez.a > 0.99) break;
        vec3 pos = ro + t*rd;
        vec2 mpv = map(pos);
        float den = clamp(mpv.x-0.3,0.,1.)*1.12;
        float dn = clamp((mpv.x + 2.),0.,3.);
        vec4 col = vec4(0);
        if (mpv.x > 0.6) {
            col = vec4(sin(vec3(5.,0.4,0.2) + mpv.y*0.1 +sin(pos.z*0.4)*0.5 + 1.8)*0.5 + 0.5, 0.08);
            col *= den*den*den;
            col.rgb *= linstep(4.,-2.5, mpv.x)*2.3;
            float dif = clamp((den - map(pos+.8).x)/9., 0.001, 1.);
            dif += clamp((den - map(pos+.35).x)/2.5, 0.001, 1.);
            col.xyz *= den*(vec3(0.005,.045,.075) + 1.5*vec3(0.033,0.07,0.03)*dif);
        }
        float fogC = exp(t*0.2 - 2.2);
        col.rgba += vec4(0.06,0.11,0.11, 0.1)*clamp(fogC-fogT, 0., 1.);
        fogT = fogC;
        rez = rez + col*(1. - rez.a);
        t += clamp(0.5 - dn*dn*.05, 0.09, 0.3);
    }
    return clamp(rez, 0.0, 1.0);
}
float getsat(vec3 c) {
    float mi = min(min(c.x, c.y), c.z);
    float ma = max(max(c.x, c.y), c.z);
    return (ma - mi)/(ma+ 1e-7);
}
vec3 iLerp(in vec3 a, in vec3 b, in float x) {
    vec3 ic = mix(a, b, x) + vec3(1e-6,0.,0.);
    float sd = abs(getsat(ic) - mix(getsat(a), getsat(b), x));
    vec3 dir = normalize(vec3(2.*ic.x - ic.y - ic.z, 2.*ic.y - ic.x - ic.z, 2.*ic.z - ic.y - ic.x));
    float lgt = dot(vec3(1.0), ic);
    float ff = dot(dir, normalize(ic));
    ic += 1.5*dir*sd*ff*lgt;
    return clamp(ic,0.,1.);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 q = fragCoord.xy/iResolution.xy;
    vec2 p = (fragCoord.xy - 0.5*iResolution.xy)/iResolution.y;
    bsMo = (iMouse.xy - 0.5*iResolution.xy)/iResolution.y;
    float time = iTime*3.;
    vec3 ro = vec3(0,0,time);
    ro += vec3(sin(iTime)*0.5, 0., 0);
    float dspAmp = .85;
    ro.xy += disp(ro.z)*dspAmp;
    float tgtDst = 3.5;
    vec3 target = normalize(ro - vec3(disp(time + tgtDst)*dspAmp, time + tgtDst));
    ro.x -= bsMo.x*2.;
    vec3 rightdir = normalize(cross(target, vec3(0,1,0)));
    vec3 updir = normalize(cross(rightdir, target));
    rightdir = normalize(cross(updir, target));
    vec3 rd = normalize((p.x*rightdir + p.y*updir)*1. - target);
    rd.xy *= rot(-disp(time + 3.5).x*0.2 + bsMo.x);
    prm1 = smoothstep(-0.4, 0.4, sin(iTime*0.3));
    vec4 scn = render(ro, rd, time);
    vec3 col = scn.rgb;
    col = iLerp(col.bgr, col.rgb, clamp(1.-prm1,0.05,1.));
    col = pow(col, vec3(.55,0.65,0.6))*vec3(1.,.97,.9);
    col *= pow(16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.12)*0.7+0.3;
    fragColor = vec4(col, 1.0);
}
"""

shader4_src = """
mat3 m = mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);
float hash(float n) { return fract(sin(n) * 43758.5453); }
float noise(in vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    return mix(mix(mix(hash(n), hash(n+1.0), f.x),
                   mix(hash(n+57.0), hash(n+58.0), f.x), f.y),
               mix(mix(hash(n+113.0), hash(n+114.0), f.x),
                   mix(hash(n+170.0), hash(n+171.0), f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float f;
    f  = 0.5000 * noise(p); p = m * p * 2.02;
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
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);
    vec2 mo = vec2(iTime * 0.1, cos(iTime * 0.25) * 3.0);
    vec3 ta = vec3(0.0, 1.0, 0.0);
    vec3 ro = 25.0 * normalize(vec3(cos(2.75 - 3.0*mo.x), 0.7 - (mo.y-1.0), sin(2.75 - 3.0*mo.x)));
    mat3 c = camera(ro, ta);
    vec3 dir = c * normalize(vec3(uv, 1.3));
    const int N = 64;
    float zstep = 40.0 / float(N);
    vec3 p = ro;
    float T = 1.0;
    vec4 color = vec4(0.0);
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
    color.rgb += mix(vec3(0.3, 0.1, 0.8), vec3(0.7, 0.7, 1.0), 1.0 - (uv.y + 1.0) * 0.5);
    color.a = 1.0;
    fragColor = color;
}
"""

ALL_SHADERS = [shader1_src, shader2_src, shader3_src, shader4_src]

# ── GLSL wrapper ────────────────────────────────────────────────

VERTEX = """
#version 330
in vec2 in_position;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

def make_frag(body):
    return f"""
#version 330
out vec4 fragColor;
uniform vec2  iResolution;
uniform float iTime;
uniform vec2  iMouse;

{body}

void main() {{
    mainImage(fragColor, gl_FragCoord.xy);
}}
"""

# ── Application ─────────────────────────────────────────────────

class App:
    def __init__(self):
        self.width, self.height = 1280, 720
        
        # Create pyglet window directly
        config = pyglet.gl.Config(
            double_buffer=True,
            major_version=3,
            minor_version=3,
        )
        self.window = pyglet.window.Window(
            width=self.width, height=self.height,
            caption="Shader Test | Loading...",
            config=config,
            resizable=True,
        )
        
        # Create moderngl context from the existing OpenGL context
        self.ctx = moderngl.create_context()
        
        # Full-screen quad
        import struct
        vertices = struct.pack('8f',
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        )
        vbo = self.ctx.buffer(vertices)
        
        # Compile all shaders
        self.programs = []
        for i, src in enumerate(ALL_SHADERS):
            try:
                prog = self.ctx.program(
                    vertex_shader=VERTEX,
                    fragment_shader=make_frag(src),
                )
                self.programs.append(prog)
                print(f"  [OK] {SHADER_NAMES[i]}")
            except Exception as e:
                print(f"  [FAIL] {SHADER_NAMES[i]}:\n    {e}")
                self.programs.append(None)
        
        # Create VAOs for each program
        self.vaos = []
        for prog in self.programs:
            if prog is not None:
                vao = self.ctx.vertex_array(prog, [(vbo, '2f', 'in_position')])
                self.vaos.append(vao)
            else:
                self.vaos.append(None)
        
        self.current = 0
        self.mouse = (0.0, 0.0)
        self.start_time = _time.perf_counter()
        
        # FPS tracking
        self.frame_count = 0
        self.fps_time = _time.perf_counter()
        self.fps = 0
        
        # ── Wire up pyglet events directly ──
        @self.window.event
        def on_draw():
            self._render()
        
        @self.window.event
        def on_key_press(symbol, modifiers):
            print(f"KEY PRESS: {symbol}")
            if symbol == pyglet.window.key.ESCAPE:
                self.window.close()
            elif symbol == pyglet.window.key.SPACE:
                self._next_shader()
            elif symbol == pyglet.window.key.LEFT:
                self._prev_shader()
            elif symbol == pyglet.window.key.RIGHT:
                self._next_shader()
        
        @self.window.event
        def on_mouse_press(x, y, button, modifiers):
            print(f"MOUSE PRESS at ({x},{y}) button={button}")
            if button == pyglet.window.mouse.LEFT:
                self._next_shader()
            elif button == pyglet.window.mouse.RIGHT:
                self._prev_shader()
        
        @self.window.event
        def on_mouse_motion(x, y, dx, dy):
            self.mouse = (float(x), float(y))
        
        @self.window.event
        def on_resize(width, height):
            self.width = width
            self.height = height
            self.ctx.viewport = (0, 0, width, height)
        
        self._update_title()
        print(f"\nLoaded {sum(1 for p in self.programs if p)} / {len(ALL_SHADERS)} shaders.")
        print("Controls: SPACE / Left-Click = next | Arrow keys / Right-Click = prev | ESC = quit\n")
    
    def _next_shader(self):
        for _ in range(len(self.programs)):
            self.current = (self.current + 1) % len(self.programs)
            if self.programs[self.current] is not None:
                break
        self._update_title()
        print(f">> Switched to {SHADER_NAMES[self.current]}")
    
    def _prev_shader(self):
        for _ in range(len(self.programs)):
            self.current = (self.current - 1) % len(self.programs)
            if self.programs[self.current] is not None:
                break
        self._update_title()
        print(f">> Switched to {SHADER_NAMES[self.current]}")
    
    def _update_title(self):
        self.window.set_caption(
            f"{SHADER_NAMES[self.current]} | FPS: {self.fps} | SPACE/Click=toggle"
        )
    
    def _render(self):
        # FPS
        self.frame_count += 1
        now = _time.perf_counter()
        if now - self.fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_time = now
            self._update_title()
        
        t = now - self.start_time
        
        self.ctx.clear(0.0, 0.0, 0.0)
        
        prog = self.programs[self.current]
        vao = self.vaos[self.current]
        if prog and vao:
            if 'iTime' in prog:
                prog['iTime'].value = t
            if 'iResolution' in prog:
                prog['iResolution'].value = (float(self.width), float(self.height))
            if 'iMouse' in prog:
                prog['iMouse'].value = self.mouse
            vao.render(moderngl.TRIANGLE_STRIP)
    
    def run(self):
        pyglet.app.run()


if __name__ == '__main__':
    print("=" * 50)
    print("  Shadertoy Performance Tester")
    print("=" * 50)
    app = App()
    app.run()

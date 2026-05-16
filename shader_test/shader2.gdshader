shader_type canvas_item;

vec3 palette(float d){
	return mix(vec3(0.2,0.7,0.9),vec3(1.,0.,1.),d);
}

vec2 rotate(vec2 p,float a){
	float c = cos(a);
    float s = sin(a);
    return p*mat2(vec2(c,s),vec2(-s,c));
}

float map(vec3 p){
    for( int i = 0; i<8; ++i){
        float t = TIME*0.2;
        p.xz =rotate(p.xz,t);
        p.xy =rotate(p.xy,t*1.89);
        p.xz = abs(p.xz);
        p.xz-=.5;
	}
	return dot(sign(p),p)/5.;
}

vec4 rm (vec3 ro, vec3 rd){
    float t = 0.;
    vec3 col = vec3(0.);
    float d;
    for(float i =0.; i<64.; i++){
		vec3 p = ro + rd*t;
        d = map(p)*.5;
        if(d<0.02){
            break;
        }
        if(d>100.){
        	break;
        }
        col+=palette(length(p)*.1)/(400.*(d));
        t+=d;
    }
    return vec4(col,1./(d*100.));
}

void fragment()
{
    vec2 iResolution = 1.0 / SCREEN_PIXEL_SIZE;
    vec2 uv = (FRAGCOORD.xy - (iResolution.xy / 2.)) / iResolution.x;
	vec3 ro = vec3(0., 0., -50.);
    ro.xz = rotate(ro.xz, TIME);
    vec3 cf = normalize(-ro);
    vec3 cs = normalize(cross(cf, vec3(0., 1., 0.)));
    vec3 cu = normalize(cross(cf, cs));
    
    vec3 uuv = ro + cf * 3. + uv.x * cs + uv.y * cu;
    
    vec3 rd = normalize(uuv - ro);
    
    vec4 col = rm(ro, rd);
    
    COLOR = col;
    COLOR.a = 1.0;
}

#version 330

in vec2 fragTexCoord;
in vec4 fragColor;
out vec4 finalColor;

uniform sampler2D texture0;
uniform vec2 resolution;
// Screen center, signed pixel radius, remaining life (negative for steady pull fields).
uniform vec4 blasts[16];
uniform float amplitudes[16];
// Nonzero makes a source a twirl (portals): the scene rotates around the
// center, most at the center and none at the radius; 1 turns it clockwise
// on screen and -1 counterclockwise. Its amplitude is the peak twist in
// radians, and it isn't limited by the pixel cap.
uniform float swirls[16];
// Pixel cap, relative wave width, lifetime fade exponent (from settings.h).
uniform vec3 distortionSettings;
uniform int blastCount;
uniform vec4 blackHoleTint;

void main()
{
    vec2 pixel = fragTexCoord * resolution;
    vec2 offset = vec2(0.0);
    vec2 swirlOffset = vec2(0.0);
    float tintCoverage = 0.0;
    for (int i = 0; i < 16; ++i) {
        if (i >= blastCount) break;
        vec2 delta = pixel - blasts[i].xy;
        float distance = length(delta);
        float radius = abs(blasts[i].z);
        float remaining = blasts[i].w;
        if (swirls[i] != 0.0) {
            float r = distance / max(radius, 1.0);
            if (r < 1.0) {
                // twist in quickly, then unwind slowly
                float progress = 1.0 - remaining;
                float falloff = (1.0 - r) * (1.0 - r);
                // sampling from a counterclockwise rotation shows the scene
                // turned clockwise, as pixel coordinates have y up
                float twist = swirls[i] * amplitudes[i] * sin(3.14159265 * sqrt(progress)) * falloff;
                float c = cos(twist);
                float s = sin(twist);
                swirlOffset += vec2(c * delta.x - s * delta.y, s * delta.x + c * delta.y) - delta;
            }
            continue;
        }
        if (remaining < 0.0) {
            // Smooth inward lens across the pull area; no moving wave or outline.
            float r = clamp(distance / max(radius, 1.0), 0.0, 1.0);
            float life = -remaining;
            float pull = 6.75 * r * (1.0 - r) * (1.0 - r) * life;
            tintCoverage += (1.0 - smoothstep(0.0, 1.0, r)) * life;
            // Sample outward so scene features appear drawn toward the projectile.
            offset += delta / max(distance, 1.0) * amplitudes[i] * pull;
            continue;
        }
        float travel = 1.0 - remaining * remaining * remaining;
        float wave = blasts[i].z < 0.0 ? 0.85 - 0.75 * travel : 0.15 + 0.7 * travel;
        float normalizedDistance = distance / max(radius, 1.0);
        float ripple = (normalizedDistance - wave) / distortionSettings.y;
        float envelope = exp(-ripple * ripple * 1.5);
        // Fade before the fixed radius outline; never warp the edge of the blast.
        float interior = 1.0 - smoothstep(0.8, 0.95, normalizedDistance);
        // A broad swell bends grid lines visibly, including at the wave crest.
        float amplitude = amplitudes[i] * pow(remaining, distortionSettings.z);
        float strength = cos(ripple * 2.0) * envelope * interior * amplitude;
        offset += delta / max(distance, 1.0) * strength * sign(blasts[i].z);
    }
    offset *= min(1.0, distortionSettings.x / max(length(offset), 0.001));
    offset += swirlOffset;
    vec2 texel = 0.5 / resolution;
    vec2 uv = clamp(fragTexCoord + offset / resolution, texel, vec2(1.0) - texel);
    vec4 scene = texture(texture0, uv) * fragColor;
    scene.rgb = mix(scene.rgb, blackHoleTint.rgb, blackHoleTint.a * min(tintCoverage, 1.0));
    finalColor = scene;
}

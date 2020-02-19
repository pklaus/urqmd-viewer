#!/usr/bin/env python

"""
(c) 2020, Philipp Klaus
(c) 2014, Aurore Deschildre, Gael Goret, Cyrille Rossant, Nicolas P. Rougier.

shader code adapted from Vispy's example molecular_viewer.py:
https://github.com/vispy/vispy/blob/master/examples/demo/gloo/molecular_viewer.py

Distributed under the terms of the new BSD License.
"""

# external dependencies:
from vispy import app, gloo, visuals
from vispy.util.transforms import perspective, translate, rotate
from vispy.visuals.transforms import STTransform, MatrixTransform
import numpy as np
import attr
from urqmd_tools.pids import LOOKUP_TABLE, ALL_SORTED
from urqmd_tools.parser.f14 import F14_Parser, Particle

from statistics import mean, stdev
import collections
import argparse
import sys
import os
import time
import pickle

r = np.random.RandomState(1237+80)

VERT_SHADER = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_light_position;
uniform vec3 u_light_spec_position;
uniform float u_aspect;

attribute vec3  a_position;
attribute vec3  a_color;
attribute float a_radius;

varying vec3  v_color;
varying vec4  v_eye_position;
varying float v_radius;
varying vec3  v_light_direction;

void main (void) {
    v_radius = a_radius;
    v_color = a_color;

    v_eye_position = u_view * u_model * vec4(a_position,1.0);
    v_light_direction = normalize(u_light_position);
    float dist = length(v_eye_position.xyz);

    gl_Position = u_projection * v_eye_position;

    // stackoverflow.com/questions/8608844/...
    //  ... resizing-point-sprites-based-on-distance-from-the-camera
    vec4  proj_corner = u_projection * vec4(a_radius, a_radius, v_eye_position.z, v_eye_position.w);  // # noqa
    gl_PointSize = 512.0 * proj_corner.x / proj_corner.w * u_aspect;
    // attempt to make very far points slighly bigger:
    //gl_PointSize = 128.0 * log(proj_corner.x / proj_corner.w * 5 + 1);
}
"""

FRAG_SHADER = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_light_position;
uniform vec3 u_light_spec_position;

varying vec3  v_color;
varying vec4  v_eye_position;
varying float v_radius;
varying vec3  v_light_direction;
void main()
{
    // r^2 = (x - x0)^2 + (y - y0)^2 + (z - z0)^2
    vec2 texcoord = gl_PointCoord* 2.0 - vec2(1.0);
    float x = texcoord.x;
    float y = texcoord.y;
    float d = 1.0 - x*x - y*y;
    if (d <= 0.0)
        discard;

    float z = sqrt(d);
    vec4 pos = v_eye_position;
    pos.z += v_radius*z;
    vec3 pos2 = pos.xyz;
    pos = u_projection * pos;
//    gl_FragDepth = 0.5*(pos.z / pos.w)+0.5;
    vec3 normal = vec3(x,y,z);
    float diffuse = clamp(dot(normal, v_light_direction), 0.0, 1.0);

    // Specular lighting.
    vec3 M = pos2.xyz;
    vec3 O = v_eye_position.xyz;
    vec3 L = u_light_spec_position;
    vec3 K = normalize(normalize(L - M) + normalize(O - M));
    // WARNING: abs() is necessary, otherwise weird bugs may appear with some
    // GPU drivers...
    float specular = clamp(pow(abs(dot(normal, K)), 40.), 0.0, 1.0);
    vec3 v_light = vec3(1., 1., 1.);
    gl_FragColor.rgba = vec4(.15*v_color + .55*diffuse * v_color
                        + .35*specular * v_light, 1.0);
}
"""



class HICCanvas(app.Canvas):

    pid_colors = {urqmdpid.id: r.uniform(low=0.2, high=1.0, size=4) for urqmdpid in ALL_SORTED}

    def __init__(self, pts, ts, b=7, a=23, fmps=2, cb=0.9224028, bb=0.0, sf=10, w=1920, h=1080, t='dark', c='by_pid', win=False):
        """
        pts: particles at different timesteps
        ts: list of the timesteps (in fm/c)
        b: before
        a: after
        fmps: fm per second (relation speed of light to visualization time)
        cb: CMS beta
        bb: boost beta
        sf: scaling factor
        w: viewport width
        h: viewport height
        t: theme ('bright' or 'dark')
        c: coloring scheme ('by_kind' or 'by_pid')
        win: start in windowed mode instead of full-screen (use F11 to toggle during run)
        """

        start = time.time()

        self.pts = pts # the particles in the time evolution
        self.ts = ts # the timesteps for self.pts
        self.b = b # amount of s before t0
        self.a = a # amount of s after t0
        self.fmps = fmps # fm / s
        self.cb = cb # cms beta
        self.bb = bb # boost beta
        self.sf = sf # scaling factor
        self.w = w # view width
        self.h = h # view height
        self.theme = t
        self.coloring = c
        self.windowed = win

        if self.theme not in ('bright', 'dark'):
            raise NotImplementedError('theme: %s' % self.theme)
        if self.coloring not in ('by_kind', 'by_pid'):
            raise NotImplementedError('coloring', self.coloring)

        self.print_current_particles = False
        self.print_current_particles_without_nucleons = True

        self.recent_fps = []
        self.last_update = 0.0
        self.last_stats_output = 0.0

        app.Canvas.__init__(self, title='UrQMD Viewer',
                            keys='interactive', size=(w, h))

        # Create program
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        self.ps = self.pixel_scale

        n = max([len(ps) for ps in self.pts])
        self.n = n
        print("maximum number of particles =", n)
        self.particles = np.zeros(n, [('a_position', 'f4', 3),
                                      ('a_color', 'f4', 4),
                                      ('a_radius', 'f4')])

        #self.particles['a_position'] = np.random.uniform(-20, +20, (n, 3)) + 1000
        self.particles['a_position'] = 1000000, 1000000, 1000000
        self.particles['a_radius'] = self.sf * self.pixel_scale
        self.particles['a_color'] = 1, 1, 1, 0

        self.translate = 40
        self.view = translate((0, 0, -self.translate))
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.theta = 0
        self.phi = 0

        # Time
        self._t = time.time()

        # Bind vertex buffers
        self.program.bind(gloo.VertexBuffer(self.particles))

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_aspect'] = w/h
        self.program['u_light_position'] = 0., 0., 2.
        self.program['u_light_spec_position'] = -5., 5., -5.

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.start = time.time()

        self.paused = False
        self.update_required = True
        self.pause_started = 0.0

        self.visuals = []
        if self.theme == 'bright':
            text_color = (0, 0, 0, 1)
        if self.theme == 'dark':
            text_color = (1, 1, 1, 1)
        self.fps_fmt = '{:.1f} FPS'
        llt = visuals.TextVisual(self.fps_fmt.format(0.0), bold=False,
            pos=[10, self.physical_size[1] - 10 - 20 - 10], color=text_color, anchor_x='left', anchor_y='bottom',
            method='gpu', font_size=10)
        self.lower_left_text = llt
        self.visuals.append(llt)
        self.time_fmt = 'τ = {:.1f} fm/c'
        ult = visuals.TextVisual(self.time_fmt.format(0.0), bold=True,
            pos=[10, 10], color=text_color, anchor_x='left', anchor_y='bottom',
            method='gpu', font_size=20)
        self.upper_left_text = ult
        self.visuals.append(ult)
        #urt = visuals.TextVisual('© Philipp Klaus', bold=True,
        urt = visuals.TextVisual('UrQMD Viewer by @pklaus', bold=True,
            pos=[self.physical_size[0] - 10, 10], color=text_color, anchor_x='right',
            anchor_y='bottom', method='gpu', font_size=20)
        self.upper_right_text = urt
        self.visuals.append(urt)
        line_pos = np.array([[0, 0, -100],
                             [0, 0, 100],
                            ])
        ba = visuals.LineVisual(pos=line_pos, color=(.5, .5, .5, 1),
            width=1, method='gl', antialias=False)
        self.beam_axis = ba
        self.visuals.append(ba)
        beam_direction = visuals.ArrowVisual(pos=np.array([[0, 0, 0], [0, 0, 1]]),
            color=(.5, .5, .5, 1), width=1, method='gl', arrow_size=0.6)
        self.visuals.append(beam_direction)
        sphere = visuals.SphereVisual(radius=0.08)
        self.visuals.append(sphere)

        self.update_model()

        self.update_projection()

        if self.theme == 'dark':
            gloo.set_state(depth_test=True, clear_color=(0, 0, 0, 1))
        elif self.theme == 'bright':
            gloo.set_state(depth_test=True, clear_color=(1, 1, 1, 1))
        # translucent particles:
        gloo.set_state(blend=True, blend_func=('src_alpha', 'one'))
        #gloo.wrappers.set_depth_range(near=-1000.0, far=10000.0)

        if not self.windowed:
            self.fullscreen = True
        self.show()
        end = time.time()

    def on_key_press(self, event):
        if event.text == ' ':
            self.paused = ~self.paused
            if self.paused:
                self.pause_started = time.time()
            else:
                self.start += time.time() - self.pause_started
        elif event.key in ('j', 'Left'):
            self.start += 0.05
        elif event.key in ('k', 'Right'):
            self.start -= 0.05
        elif event.key in ('h', 'PageDown'):
            self.start += 0.5
        elif event.key in ('l', 'PageUp'):
            self.start -= 0.5
        elif event.key in ('e',):
            self.sf *= 1.1
        elif event.key in ('q',):
            self.sf *= 0.9
        elif event.key in ('p',):
            self.print_current_particles = True
        else:
            print(event.key)
        self.update_required = True
        self.update()

    def on_resize(self, event):
        print(event.size, event.physical_size)
        vp = (0, 0, *event.physical_size)
        gloo.set_viewport(*vp)
        #self.context.set_viewport(*vp)
        for v in (v for v in self.visuals if type(v) is visuals.TextVisual):
            v.transforms.configure(canvas=self, viewport=vp)
        self.upper_right_text.pos = [event.size[0] - 10, 10]
        self.lower_left_text.pos = [10, event.size[1] - 10 - 10 - 20]
        self.update_projection(*event.size)
        self.update()

    def update_projection(self, width=None, height=None):
        if width is None or height is None:
            width, height = self.size
        self.projection = perspective(fovy=40.0, aspect=width/height, znear=5.0, zfar=500.0)
        self.program['u_projection'] = self.projection
        self.program['u_aspect'] = width/height

    def on_mouse_wheel(self, event):
        self.translate -= event.delta[1]
        #self.translate = max(-1, self.translate)
        self.view = translate((0, 0, -self.translate))

        self.program['u_view'] = self.view
        self.update()

    def _normalize(self, x_y):
        x, y = x_y
        w, h = float(self.size[0]), float(self.size[1])
        return x/(w/2.)-1., y/(h/2.)-1.

    def on_mouse_move(self, event):
        if event.is_dragging:
            x0, y0 = self._normalize(event.press_event.pos)
            x1, y1 = self._normalize(event.last_event.pos)
            x, y = self._normalize(event.pos)
            dx, dy = x - x1, -(y - y1)
            button = event.press_event.button
            self.phi += dx*150
            self.theta -= dy*150
            self.update_model()
            self.update()
            #if button == 1:
            #    pass
            #elif button == 2:
            #    pass

    def on_draw(self, event):
        gloo.clear()
        # need to do this every time as the texts overwrites it:
        for v in (v for v in self.visuals if type(v) is not visuals.TextVisual):
            pass#v.draw()
        gloo.set_state(depth_test=True)
        self.program.draw('points')
        for v in (v for v in self.visuals if type(v) is visuals.TextVisual):
            v.draw()

    def update_model(self):
        mt = MatrixTransform()
        mt.rotate(120, (1, 1, 1))
        mt.rotate(self.phi, (0, 1, 0))
        mt.rotate(self.theta, (1, 0, 0))
        self.beam_axis.transform = mt
        self.model = mt.matrix
        self.program['u_model'] = self.model

    def stats_output(self):
        if len(self.recent_fps) < 2: return
        if (time.time() - self.last_stats_output) > 0.5:
            self.last_stats_output = time.time()
            fps, self.recent_fps = self.recent_fps, []
            self.lower_left_text.text = self.fps_fmt.format(mean(fps))


    def update(self):
        super().update()
        self.recent_fps.append(1/(time.time() - self.last_update))
        self.last_update = time.time()
        self.stats_output()

    def on_timer(self, event):
        if self.paused and not self.update_required:
            return
        self.update_required = False

        if self.paused:
            now = self.pause_started
        else:
            now = time.time()
        t = (now - self.start) % (self.a + self.b) - self.b
        t_fm = t * self.fmps # time in fm before (-) or after (+) the collision
        self.upper_left_text.text = self.time_fmt.format(t_fm)

        white = (1, 1, 1, 1)
        red = [1, 0.3, 0.3, 1]
        green = [.2, .5, 1, 1]
        yellow = [.8, 1, .2, 1]
        blue = [.1, 0.1, .9, 1]

        baryons = [i for i in range(1, 56)]
        baryons += [-baryon for baryon in baryons]
        mesons = [101, 106, 102, 107, 104, 108, 103, 109, 111, 110, 105, 112, 114, 113, 115, 116, 122, 121, 123, 124, 118, 117, 119, 120, 126, 125, 127, 128, 130, 129, 131, 132]
        mesons += [-meson for meson in mesons]


        if t <= self.ts[0]:
            t_fm_0 = self.ts[0] # initial timestep
            ps = self.pts[0]
            #self.particles = np.zeros(len(ps), [('a_position', 'f4', 3),
            #                              ('a_color', 'f4', 4),
            #                              ('a_radius', 'f4')])
            self.particles['a_position'][0:len(ps)] = np.array([[p.rx, p.ry, p.rz]
                                                                for p in ps])
            self.particles['a_position'][:,2] += np.where(
                self.particles['a_position'][:,2] < 0,
                (t_fm-t_fm_0)*( self.cb+self.bb)/(1+self.cb*self.bb),
                (t_fm-t_fm_0)*(-self.cb+self.bb)/(1+self.cb*self.bb)
            )
            # move everything else away...
            self.particles['a_position'][len(ps):] = 100000, 100000, 100000
        else:
            # find the best suiting timestamp for interpolation:
            for i, ts_fm in enumerate(self.ts):
                if t_fm <= ts_fm:
                    break
            t_fm_0 = ts_fm
            d_t_fm = t_fm - t_fm_0
            # fetch the particle set belonging to that timestep
            ps = self.pts[i]
            #self.particles = np.zeros(len(ps), [('a_position', 'f4', 3),
            #                              ('a_color', 'f4', 4),
            #                              ('a_radius', 'f4')])
            self.particles['a_position'][0:len(ps)] = np.array([[p.rx, p.ry, p.rz] for p in ps]) +\
               + d_t_fm * np.array([p.beta3 for p in ps])

        if self.print_current_particles:
            if self.print_current_particles_without_nucleons:
                print_ps = [p for p in ps if not (p.id == 1 and p.ncl == 0)]
                print("Particles currently in the scene (without any nucleons that didn't collide so far):")
            else:
                print_ps = ps
                print("Particles currently in the scene:")
            pids = [p.id for p in print_ps]
            for pid, freq in collections.Counter(pids).most_common():
                print("  ", LOOKUP_TABLE[pid].name, " - amount currently in the scene:", freq)
            self.print_current_particles = False

        # DEBUG Output of the xyz extents of the collision
        # (outermost particle positions)
        #x_extent = tuple(func(self.particles['a_position'][0:len(ps)][0,:]) for func in (np.amin, np.amax))
        #y_extent = tuple(func(self.particles['a_position'][0:len(ps)][0,:]) for func in (np.amin, np.amax))
        #z_extent = tuple(func(self.particles['a_position'][0:len(ps)][0,:]) for func in (np.amin, np.amax))
        #print(x_extent, y_extent, z_extent)

        radius_from = 'm0' #m0'
        if radius_from == 'm':
            self.particles['a_radius'][0:len(ps)] = np.array([p.m**(1/3) for p in ps])
        elif radius_from == 'm0':
            self.particles['a_radius'][0:len(ps)] = np.array([p.m0**(1/3) for p in ps])
        elif radius_from in ('E', 'Ekin'):
            self.particles['a_radius'][0:len(ps)] = np.array([p.E**(1/3) for p in ps])
        self.particles['a_radius'][0:len(ps)] *= self.sf * self.pixel_scale
        #self.particles['a_radius'][0:len(ps)][[p.id == 1 for p in ps]] = 0.5

        # make any particles white that are not colored otherwise
        #self.particles['a_color'][0:len(ps)] = 1, 1, 1, 1
        a_color = np.zeros((len(ps), 4))
        if self.coloring == 'by_kind':
            a_color[[p.id in mesons for p in ps]] = yellow # mesons
            a_color[[p.id in baryons for p in ps]] = green #blue # excited baryons
            a_color[[p.id == 1 for p in ps]] = red # nucleons (mostly projectile and target nucleons)
            a_color[[p.id == 1 and p.ncl > 0 for p in ps]] -= (0.1, 0.1, 0.1, 0) # collided projectile and target nucleons
        elif self.coloring == 'by_pid':
            a_color = np.array([self.pid_colors[p.id] for p in ps])
            a_color[[p.ncl == 0 for p in ps]] += (0.2, 0.2, 0.2, 0)
        self.particles['a_color'][0:len(ps)] = a_color


        #self.particles['a_color'][len(ps):self.n] = 0, 0, 0, 0
        #self.particles['a_color'][len(ps):] = 0, 0, 0, 0
        self.particles['a_position'][len(ps):] = 100000, 100000, 100000

        # special treatment for some particles
        # Φ (phi)
        phis = (109, 128, 132)
        self.particles['a_color'][0:len(ps)][[p.id in phis for p in ps]] = blue if round(t*4) % 2 else yellow
        #self.particles['a_radius'][0:len(ps)][[p.id in phis for p in ps]] = 4
        if any([p.id in phis for p in ps]):
            print("found a Φ (phi) at", t_fm)
        # 27 = Lambda, 106 = K+ / K0, 108 = K*
        if round(t_fm*4)%2:
            self.particles['a_color'][0:len(ps)][[p.id in (27, 106, 108) for p in ps]] = white

        #print("mean position", self.particles['a_position'].mean())

        self.program.bind(gloo.VertexBuffer(self.particles))
        self.update()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('urqmd_file', metavar='URQMD_FILE', type=argparse.FileType('r'), help="Must be of type .f14")
    parser.add_argument('--after', default=40, type=float)
    parser.add_argument('--before', default=5, type=float)
    parser.add_argument('--width', default=900, type=float)
    parser.add_argument('--height', default=600, type=float)
    parser.add_argument('--windowed', action='store_true')
    parser.add_argument('--cms-beta', default=0.9224028, type=float)
    parser.add_argument('--boost-beta', default=0.0, type=float)
    parser.add_argument('--fm-per-sec', default=3, type=float)
    parser.add_argument('--scaling-factor', default=1, type=float)
    parser.add_argument('--theme', choices=('bright', 'dark'), default='dark')
    parser.add_argument('--coloring-scheme', choices=('by_kind', 'by_pid'), default='by_pid')
    args = parser.parse_args()

    start_loading_data = time.time()
    cache_file = '.cache.'+os.path.basename(args.urqmd_file.name)+'.pickle'
    try:
        print("Trying to open a cached version of the .f14 output")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            pts = data['pts']
            ts = data['ts']
    except:
        print("Cached version unavailable, now parsing the .f14 file...")
        for event in F14_Parser(args.urqmd_file).get_events():
            particles = [Particle(particle_properties) for particle_properties in event['particle_properties']]
            ts = sorted(list(set(p.time for p in particles)))
            pts = []
            # naive approach:
            for i, t in enumerate(ts):
                print(i, "out of", len(ts), "filtered")
                selection = []
                for p in particles:
                    if p.time == t:
                        selection.append(p)
                    if p.time > t:
                        break
                for p in selection:
                    particles.remove(p)
                pts.append(selection)
            break # only read the very first event in the file
        with open(cache_file, 'wb') as f:
            pickle.dump({'ts': ts, 'pts': pts}, f, pickle.HIGHEST_PROTOCOL)
    print("Done loading particle data after {:.3f} s".format(time.time() - start_loading_data))

    if args.boost_beta:
        for ps in pts:
            for p in ps:
                p.boost(args.boost_beta)
        # The Lorentz boost changes our timesteps, too...
        # actually now every boosted particle has a different timestamp, so this is broken...
        ts = sorted(list(set(ps[0].time for ps in PS_TS)))

    c = HICCanvas(pts,
                  ts,
                  a=args.after,
                  b=args.before,
                  cb=args.cms_beta,
                  bb=args.boost_beta,
                  fmps=args.fm_per_sec,
                  sf=args.scaling_factor,
                  w=args.width,
                  h=args.height,
                  t=args.theme,
                  c=args.coloring_scheme,
                  win=args.windowed,
                 )
    app.run()

if __name__ == '__main__':
    main()

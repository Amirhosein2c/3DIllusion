import moderngl
import numpy as np
from pyrr import Matrix44

from ported._example import Example

# importing head tracker module
from facetracking import *

class CrateExample(Example):
    title = "Crate"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tracker = HeadTracker(1)
        self.tracker.ReadCamera()

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;

                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_vert = in_position;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec3 Light;
                uniform sampler2D Texture;

                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;

                out vec4 f_color;

                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    f_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
                }
            ''',
        )

        self.mvp = self.prog['Mvp']
        self.light = self.prog['Light']

        self.scene = self.load_scene('crate.obj')
        self.vao = self.scene.root_nodes[0].mesh.vao.instance(self.prog)
        self.texture = self.load_texture_2d('crate.png')

    def render(self, time, frame_time):
        
        # angle = time
        self.tracker.ReadCamera()
        control_point = self.tracker.getDetections()
        camera_FOV_width, camera_FOV_height = self.tracker.getFrameSize()
        # rotZ = ((camera_FOV_width - (control_point[0] * 1.5))/camera_FOV_width)
        # rotY = ((- camera_FOV_height + (control_point[1] * 1.5))/camera_FOV_height)
        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # camera_pos = (np.cos(rotZ) * 3.0, np.sin(rotY) * 3.0, 2.0)
        camera_pos = (7.0, 0.0, 0.5)

        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 100.0)
        lookat = Matrix44.look_at(
            camera_pos,
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 1.0),
        )

        rotateZ = Matrix44.from_z_rotation(((camera_FOV_width - (control_point[0] * 2))/camera_FOV_width))
        rotateY = Matrix44.from_y_rotation(((- camera_FOV_height + (control_point[1] * 2))/camera_FOV_height))

        self.mvp.write((proj * lookat * rotateY * rotateZ).astype('f4'))
        self.light.value = camera_pos
        self.texture.use()
        self.vao.render()


if __name__ == '__main__':
    CrateExample.run()

import arcade
import pymunk
import timeit
import math


SCREEN_WIDTH = 1800
SCREEN_HEIGHT = 800
SCREEN_TITLE = "Physica Generation"


class PhysicsSprite(arcade.Sprite):
    def __init__(self, pymunk_shape, filename):
        super().__init__(filename, center_x=pymunk_shape.body.position.x, center_y=pymunk_shape.body.position.y)
        self.pymunk_shape = pymunk_shape


class CircleSprite(PhysicsSprite):
    def __init__(self, pymunk_shape, filename):
        super().__init__(pymunk_shape, filename)
        self.width = pymunk_shape.radius * 2
        self.height = pymunk_shape.radius * 2


class BoxSprite(PhysicsSprite):
    def __init__(self, pymunk_shape, filename, width, height):
        super().__init__(pymunk_shape, filename)
        self.width = width
        self.height = height


def make_body(env, name, x, y, friction = 43.9, mass = 30.0, box_shape = None):
    
    if box_shape is None:
        box_shape = (32,32)
    moment = pymunk.moment_for_box(mass, box_shape)
    body = pymunk.Body(mass, moment)
    body.position = pymunk.Vec2d(x, y)
    shape = pymunk.Poly.create_box(body, box_shape)
    shape.elasticity = 0.001
    shape.friction = 43.9
    env.space.add(body, shape)

    sprite = BoxSprite(shape, "/Users/melkor/Documents/datasets/PatchWork/{}".format(name), width=box_shape[0], height=box_shape[1])
    env.sprite_list.append(sprite)

class PhysicaEnv(arcade.Window):
    """ Main application class. """

    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        #arcade.set_background_color((61,77,86))
        arcade.set_background_color((1,47,65))
        self.paused = False
        self.hold = 0
        
        # -- Pymunk
        self.space = pymunk.Space()
        self.space.iterations = 35
        self.space.gravity = (0.0, -900.0)

        # Lists of sprites or lines
        self.sprite_list: arcade.SpriteList[PhysicsSprite] = arcade.SpriteList()
        self.static_lines = []

        # Used for dragging shapes around with the mouse
        self.shape_being_dragged = None
        self.last_mouse_position = 0, 0

        self.draw_time = 0
        self.processing_time = 0

        # Create the floor
        floor_height = 5
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, [0, floor_height], [SCREEN_WIDTH, floor_height], 1.0)
        shape.friction = 20

        self.space.add(shape, body)
        self.static_lines.append(shape)

        self.data = []

        size = 32


    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        self.clear()

        # Start timing how long this takes
        draw_start_time = timeit.default_timer()

        # Draw all the sprites
        self.sprite_list.draw()

        # Draw the lines that aren't sprites
        for line in self.static_lines:
            body = line.body

            pv1 = body.position + line.a.rotated(body.angle)
            pv2 = body.position + line.b.rotated(body.angle)
            arcade.draw_line(pv1.x, pv1.y, pv2.x, pv2.y, (39,45,52), 2)


        # Display timings
        output = f"Processing time: {self.processing_time:.3f}"
        arcade.draw_text(output, 20, SCREEN_HEIGHT - 20, arcade.color.WHITE, 12)

        arcade.draw_text("+", self.graber_x,self.graber_y, arcade.color.WHITE, 12)

        output = f"Drawing time: {self.draw_time:.3f}"
        arcade.draw_text(output, 20, SCREEN_HEIGHT - 40, arcade.color.WHITE, 12)

        self.draw_time = timeit.default_timer() - draw_start_time
        

def render_physica():
    pass
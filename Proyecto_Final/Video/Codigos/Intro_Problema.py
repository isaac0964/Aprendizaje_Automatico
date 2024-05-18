from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from mnist_loader import load_mnist
from itertools import product, chain
import numpy as np
import Dense
from copy import deepcopy

# Load mnist data
(X_train, y_train), (X_test, y_test) = load_mnist()

class ShowCreationThenDestruction(ShowPassingFlash):
    def __init__(self, vmobject: VMobject, time_width: float = 2.0, **kwargs):
        super().__init__(vmobject, time_width=time_width, **kwargs)

class Incremento(VoiceoverScene):
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))

        title = Text("Presentación del Problema", color=YELLOW).scale(0.9).to_corner(UL)

        # Value tracker to update the scene with an x-value
        vt = ValueTracker(1950)  # Starts at 1950

        # Load Figures
        pi_concerned = SVGMobject("../imgs/concerned.svg").scale(0.8).shift(RIGHT * 3, UP*2.5)
        bug = SVGMobject("../imgs/bug.svg").scale(0.3).shift(RIGHT * 4)
        bug2 = SVGMobject("../imgs/bug.svg").scale(0.3).shift(RIGHT * 3.9, UP*0.8)
        bug3 = SVGMobject("../imgs/bug.svg").scale(0.3).shift(RIGHT * 4.8, DOWN*0.1)
        corn = ImageMobject("../imgs/corn.png").scale(0.5).rotate(20 * PI/180).shift(RIGHT*4.15).set_z_index(-999)

        # Declare x and y axes
        ax = Axes(x_range=[1950, 2022, 5], y_range=[2, 8, 0.5], axis_config={"font_size": 14}).scale(0.6).to_edge(LEFT)
        ax.add_coordinates()

        # Declare function and plot
        f = always_redraw(lambda: ax.plot(lambda x: (2.4259596881740539e13 + ((-1.0434307915781960e10) * (x**1))
             - ((2.0493102678495575e7) * (x**2)) + ((1.1759279543417633e4) * (x**3)) +((3.3468706348675488) * (x**4)) 
             + ((-3.2119788376506913e-3) * (x**5)) + (5.2729275147032127e-7 * (x**6))) * 1e-9,
             color=BLUE, x_range=[1950, vt.get_value()]))

        # Add ddot to trace the function, also pointed to the Value Tracker
        f_dot = always_redraw(lambda: Dot(point=ax.c2p(vt.get_value(), f.underlying_function(vt.get_value())), color=BLUE))

        # Add number of population above dot
        number = always_redraw(lambda: DecimalNumber(f.underlying_function(vt.get_value())).set_color(BLUE).scale(0.5).next_to(f_dot, UP))

        # Add axes labels
        labels = ax.get_axis_labels(Tex("Año").scale(0.5), Tex("Pobalción ($10^9$)").scale(0.5))

        with self.voiceover(text= "Presentar problema") as tracker:
            self.play(Write(title))
            self.wait(tracker.duration-1)
    
        # Animate Axis being drawn
        self.play(Write(ax), Write(labels))

        # Add function and trace dot
        self.add(f, f_dot, number)

        # Animate Value Tracker across n seconds, updating plots and tracing dots
        with self.voiceover(text= "Crecimiento") as tracker:
            self.play(vt.animate.set_value(2020), run_time=6)
            # Fade Out dot
            self.play(FadeOut(f_dot))
            self.add(corn)
            self.wait(tracker.duration-7)
        
        with self.voiceover(text= "Bugs") as tracker:
            self.wait(1)
            self.add(pi_concerned, bug)
            self.wait(3)
            self.play(Wiggle(bug, n_wiggles=1000))

        with self.voiceover(text= "Temprana") as tracker:  
            self.wait(5)      
            self.add(bug2, bug3)
            self.play(*[Wiggle(b, n_wiggles=1000) for b in [bug, bug2, bug3]])
            self.wait()

class ANN(VoiceoverScene):
    # Basado en https://github.com/3b1b/videos/blob/master/_2017/nn/part1.py
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))
        # Define neural network
        self.network = Dense.load_pretrained_net()
        # Get random digit as input
        inputv = (X_train[np.random.randint(len(X_train))] * 255).astype(np.uint8)

        # Draw digit with surroinding rectangle
        inputm = VGroup(*[Square(side_length=1, color=WHITE, stroke_width=0, fill_opacity = (p/255)) for p in inputv]).arrange_in_grid(28, 28, buff = 0)
        inputm.scale(0.07).to_edge(UL)
        self.play(FadeIn(SurroundingRectangle(inputm, buff=0, color=RED)))

        # Neural Network
        self.netmob = NetworkMobject((784, 16, 16, 10)).scale(0.7)
        self.netmob.save_state()
        self.add(self.netmob)
        
        with self.voiceover(text= "Avances CNN.") as tracker:
            for i in range(int(tracker.duration//8)):
                self.play(FadeIn(inputm))
                # Send input values to network
                neurnoas_iniciales = self.netmob.layers[0].neurons.copy()
                neurnoas_iniciales.set_stroke(WHITE, width=0)
                neurnoas_iniciales.set_fill(WHITE, 0)
                image_mover = VGroup(*[pixel.copy() for pixel in inputm if pixel.fill_opacity > 0.1])
                self.play(Transform(image_mover, neurnoas_iniciales, remover=True, run_time=1))
                self.feed_forward(inputv)
                self.wait(0.5)
                self.reset_display(inputm)
                # Draw digit
                inputv = (X_train[np.random.randint(len(X_train))] * 255).astype(np.uint8)
                inputm = VGroup(*[Square(side_length=1, color=WHITE, stroke_width=0, fill_opacity = (p/255)) for p in inputv]).arrange_in_grid(28, 28, buff = 0)
                inputm.scale(0.07).to_edge(UL)

            self.wait(tracker.duration/8 - tracker.duration//8)
             
    def feed_forward(self, inputv):
        activations = self.network.get_activations_of_all_layers(inputv)

        for i, activation in enumerate(activations):
            self.show_layer_activation(i, activation)

    def show_layer_activation(self, layer_idx, activation_vec):
        layer = self.netmob.layers[layer_idx]
        active_layer = self.netmob.get_active_layer(layer_idx, activation_vec)

        anims = [Transform(layer, active_layer)]

        if layer_idx > 0:
            anims += self.netmob.get_edge_propogation_animations(layer_idx-1)
        
        self.play(*anims)

    def reset_display(self, image):
        self.netmob.restore()
        self.play(FadeOut(image))
        
class RecorderExample(VoiceoverScene):
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))

        circle = Circle()
        square = Square().shift(2 * RIGHT)

        with self.voiceover(text="This circle is drawn as I speak.") as tracker:
            self.play(Create(circle), run_time=tracker.duration)

        with self.voiceover(text="Let's shift it to the left 2 units.") as tracker:
            self.play(circle.animate.shift(2 * LEFT), run_time=tracker.duration)

        with self.voiceover(text="Now, let's transform it into a square.") as tracker:
            self.play(Transform(circle, square), run_time=tracker.duration)

        with self.voiceover(text="Thank you for watching."):
            self.play(Uncreate(circle))

        self.wait()

class NetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius" : 0.15,
        "neuron_to_neuron_buff" : SMALL_BUFF,
        "layer_to_layer_buff" : 2,
        "neuron_stroke_color" : BLUE,
        "neuron_stroke_width" : 3,
        "neuron_fill_color" : GREEN,
        "edge_color" : GREY_B,
        "edge_stroke_width" : 1,
        "edge_propogation_color" : GREEN,
        "edge_propogation_time" : 1,
        "max_shown_neurons" : 16,
        "brace_for_large_layers" : False,
        "average_shown_activation_of_large_layer" : True,
        "include_output_labels" : True,
    }
    def __init__(self, layers_dims, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.__dict__.update(self.CONFIG)
        self.layer_sizes = layers_dims
        self.add_neurons()
        self.add_edges()

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size)
            for size in self.layer_sizes
        ])
        layers.arrange(RIGHT, buff = self.layer_to_layer_buff)
        self.layers = layers
        self.add(self.layers)
        if self.include_output_labels:
            self.add_output_labels()

    def get_layer(self, size):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
                radius = self.neuron_radius,
                stroke_color = self.neuron_stroke_color,
                stroke_width = self.neuron_stroke_width,
                fill_color = self.neuron_fill_color,
                fill_opacity = 0,
            )
            for x in range(n_neurons)
        ])   
        neurons.arrange(
            DOWN, buff = self.neuron_to_neuron_buff
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = Tex("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    def get_edge(self, neuron1, neuron2):
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff = self.neuron_radius,
            stroke_color = self.edge_color,
            stroke_width = self.edge_stroke_width,
        )

    def get_active_layer(self, layer_index, activation_vector):
        layer = deepcopy(self.layers[layer_index])
        self.activate_layer(layer, activation_vector)
        return layer

    def activate_layer(self, layer, activation_vector):
        n_neurons = len(layer.neurons)
        av = activation_vector
        def arr_to_num(arr):
            return (np.sum(arr > 0.1) / float(len(arr)))**(1./3)

        if len(av) > n_neurons:
            if self.average_shown_activation_of_large_layer:
                indices = np.arange(n_neurons)
                indices *= int(len(av)/n_neurons)
                indices = list(indices)
                indices.append(len(av))
                av = np.array([
                    arr_to_num(av[i1:i2])
                    for i1, i2 in zip(indices[:-1], indices[1:])
                ])
            else:
                av = np.append(
                    av[:n_neurons/2],
                    av[-n_neurons/2:],
                )
        for activation, neuron in zip(av, layer.neurons):
            neuron.set_fill(
                color = self.neuron_fill_color,
                opacity = activation
            )
        return layer

    def activate_layers(self, input_vector):
        activations = self.neural_network.get_activation_of_all_layers(input_vector)
        for activation, layer in zip(activations, self.layers):
            self.activate_layer(layer, activation)

    def deactivate_layers(self):
        all_neurons = VGroup(*chain(*[
            layer.neurons
            for layer in self.layers
        ]))
        all_neurons.set_fill(opacity = 0)
        return self

    def add_output_labels(self):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = Tex(str(n))
            label.set_height(0.75*neuron.get_height())
            label.move_to(neuron)
            label.shift(neuron.get_width()*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def get_edge_propogation_animations(self, index):
        edge_group_copy = self.edge_groups[index].copy()
        edge_group_copy.set_stroke(
            self.edge_propogation_color,
            width = 1.5*self.edge_stroke_width
        )
        return [ShowCreationThenDestruction(
            edge_group_copy, 
            run_time = self.edge_propogation_time,
            lag_ratio = 0
        )]






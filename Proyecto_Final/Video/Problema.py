from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from mnist_loader import load_mnist
from itertools import product, chain
import numpy as np

# Codigo basado en: 
# https://www.3blue1brown.com
# https://www.youtube.com/watch?v=jDe5BAsT2-Y
# https://github.com/3b1b/videos

# Load mnist data
(X_train, y_train), (X_test, y_test) = load_mnist()

ejemplo = (X_train[7] * 255).astype(np.uint8)
# Pintar el digito con cuadros en una cuadricula
digito = VGroup(*[Square(side_length=1, color=WHITE, stroke_opacity=max(.1, (p/255)), fill_opacity = (p/255)) for p in ejemplo]).arrange_in_grid(28, 28, buff = 0)
digito.save_state()
# Pintar el digito con valores
digito_vals = VGroup(*[Square(side_length=1, color=WHITE, stroke_opacity=max(.2, (p/255)), fill_opacity = (p/255) ).add(Tex('$' + str(p) + '$')) for p in ejemplo]).arrange_in_grid(28, 28, buff = 0)
digito_vals.save_state()
# Solo valores de intensidad sin digito
vals = VGroup(*[Square(side_length=1, color=WHITE, stroke_opacity=.2, fill_opacity = 0 ).add(Tex('$' + str(p) + '$')) for p in ejemplo]).arrange_in_grid(28, 28, buff = 0)
vals.save_state()

class IntroMNIST(VoiceoverScene, ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.4,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
            },
            **kwargs
        )

    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))

        # Presentar Seccion ---------------------------------------
        texto_presentacion = Text("Introducción al Problema", color=YELLOW, font_size=50)
        with self.voiceover(text="pres_problema-") as tracker:
            self.play(Write(texto_presentacion))
            self.wait(tracker.duration)
            self.play(Unwrite(texto_presentacion))

        # Presentar MNIST ------------------------------------------
        # Reiniciar valores del digito al estado inicial
        digito.restore()
        digito_vals.restore()
        vals.restore()

        # Posicionar el digito
        digito.scale(.2).move_to((-4, .2, 0))
        digito_vals.scale(.2).move_to((-4, .2, 0))
        vals.scale(.2).move_to((-4, .2, 0))

        with self.voiceover(text="intro mnist-") as tracker:
            self.wait(tracker.duration)

        # Texto del tamano
        llave_digito = Brace(digito, DOWN)
        texto_digito = Text("Tamaño: (28x28)").scale(0.6).next_to(llave_digito, DOWN)
        with self.voiceover(text="digito mnist-") as tracker:
            # Agregar el digito a la escena
            self.play(FadeIn(digito), FadeIn(SurroundingRectangle(digito, buff=0, color=WHITE)))
            self.wait(tracker.duration-1)
            # Animar el texto de tamano
            self.play(FadeIn(llave_digito), Write(texto_digito))

        with self.voiceover(text="intensiada a valor-") as tracker:
            self.wait(tracker.duration-1)
            # Convertir intensidades a valores
            self.play(ReplacementTransform(digito, vals))

        # Hacer zoom al centro de los valores en un cuadro de la image
        self.zoomed_camera.frame.set_stroke(RED)
        self.zoomed_camera.frame.set_scale(3)
        self.zoomed_camera.frame.set_z_index(3)

        # Crear Zooom ---------------------------------------
        with self.voiceover(text="intro zoom-") as tracker:
            self.play(Create(self.zoomed_camera.frame), run_time=tracker.duration/4)
            self.activate_zooming()
            self.play(self.get_zoomed_display_pop_out_animation(), run_time=tracker.duration/4)
            self.wait(tracker.duration/4)
            self.play(self.zoomed_camera.frame.animate.move_to((-4, 1, .0)), run_time=tracker.duration/4)
        digito.restore()
        digito.scale(.2).move_to((-4, .2, 0))

        with self.voiceover(text="explicar zoom-") as tracker:
            self.wait(tracker.duration)

        with self.voiceover(text="regresar imagen-") as tracker:
            self.play(FadeOut(vals), FadeIn(digito_vals), run_time=tracker.duration/5)
            self.wait(tracker.duration/5)
            self.play(FadeOut(digito_vals), FadeIn(digito), run_time=tracker.duration/5)
            self.wait(tracker.duration/5)
            self.play(self.zoomed_camera.frame.animate.shift(UP*7), FadeOut(self.zoomed_display, shift=UP*7), run_time=tracker.duration/5)

        with self.voiceover(text="Pasar a FCN-") as tracker:
            self.wait(tracker.duration)

class Prueba(Scene):
    def construct(self):
        inputv = (X_train[np.random.randint(len(X_train))] * 255).astype(np.uint8)
        # Pintar el digito con cuadros en una cuadricula
        inputm = VGroup(*[Square(side_length=1, color=WHITE, stroke_width=0, fill_opacity = (p/255)) for p in inputv]).arrange_in_grid(28, 28, buff = 0)
        inputm.scale(0.07).to_edge(UL)
        self.play(FadeIn(inputm), FadeIn(SurroundingRectangle(inputm, buff=0, color=BLUE)))
        
        netmob = NetworkMobject((784, 16, 16, 10)).scale(0.7)
        self.add(netmob)
        # Pasar los vals a la red
        neurnoas_iniciales = netmob.layers[0].neurons.copy()
        neurnoas_iniciales.set_stroke(WHITE, width=0)
        neurnoas_iniciales.set_fill(WHITE, 0)
        image_mover = VGroup(*[pixel.copy() for pixel in inputm if pixel.fill_opacity > 0.1])
        self.play(Transform(image_mover, neurnoas_iniciales, remover=True, run_time=1))

        # Feed forward

class NetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius" : 0.15,
        "neuron_to_neuron_buff" : MED_SMALL_BUFF,
        "layer_to_layer_buff" : 2,
        "neuron_stroke_color" : BLUE,
        "neuron_stroke_width" : 1,
        "neuron_fill_color" : GREEN,
        "edge_color" : GREY_B,
        "edge_stroke_width" : 0.8,
        "edge_propogation_color" : YELLOW,
        "edge_propogation_time" : 1,
        "max_shown_neurons" : 16,
        "brace_for_large_layers" : True,
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
        layer = self.layers[layer_index].deepcopy()
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
    """
    def get_edge_propogation_animations(self, index):
        edge_group_copy = self.edge_groups[index].copy()
        edge_group_copy.set_stroke(
            self.edge_propogation_color,
            width = 1.5*self.edge_stroke_width
        )
        return [ShowCreationThenDestruction(
            edge_group_copy, 
            run_time = self.edge_propogation_time,
            lag_ratio = 0.5
        )]
    """
    def add_output_labels(self):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = Tex(str(n))
            label.set_height(0.75*neuron.get_height())
            label.move_to(neuron)
            label.shift(neuron.get_width()*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

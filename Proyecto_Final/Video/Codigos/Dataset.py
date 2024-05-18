from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from mnist_loader import load_mnist
from itertools import product, chain
import numpy as np

# Load mnist data
(X_train, y_train), (X_test, y_test) = load_mnist()

ejemplo = (X_train[7] * 255).astype(np.uint8)
# Pintar el digito con cuadros en una cuadricula
digito = VGroup(*[Square(side_length=1, color=WHITE, stroke_opacity=max(.1, (p/255)), fill_opacity = (p/255)) for p in ejemplo]).arrange_in_grid(28, 28, buff = 0)
digito.save_state()

# Solo valores de intensidad sin digito
vals = VGroup(*[Square(side_length=1, color=WHITE, stroke_opacity=.2, fill_opacity = 0 ).add(Tex('$' + str(p) + '$')) for p in ejemplo]).arrange_in_grid(28, 28, buff = 0)
vals_norm = VGroup(*[Square(side_length=1, color=WHITE, stroke_opacity=.2, fill_opacity = 0 ).add(Tex('$' + str(round(p/255,2)) + '$')) for p in ejemplo]).arrange_in_grid(28, 28, buff = 0)
vals.save_state()

class datos(VoiceoverScene, ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.4,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
            },
            **kwargs)
        
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))
        # Cargar imagenes y etiquetas
        acaros = ImageMobject("imgs/acaros.jpg").scale(3)
        acaros_tex = Text("Acaros").next_to(acaros, UP)

        afidos = ImageMobject("imgs/aphids.jpg").scale(2)
        afidos_tex = Text("Áfidos").next_to(afidos, UP)

        escara = ImageMobject("imgs/escara.jpg").scale(0.9)
        escara_tex = Text("Escarabajo ampolla").next_to(escara, UP)

        b1 = ImageMobject("imgs/b1.jpg").scale(1.5)
        b2 = ImageMobject("imgs/b2.jpg").scale(0.33)
        b = Group(*[b1, b2]).arrange(LEFT)
        b_tex = Text("Gusano cortador negro").next_to(b, UP)

        tam_entrada = Text("Cambiar tamaño de las imágenes").to_corner(UL)
        norm = Text("Normalizar intensidades").to_corner(UL)

        # Define section title
        title = Text("Presentación del Conjunto de Datos (IP102)", color=RED).scale(0.9).to_edge(UL)

        num_datos = MarkupText(f'El dataset consta de <span fgcolor="{RED}">75,222</span> imágenes de\n insectos tomadas en el campo', 
                               color=WHITE).scale(0.7)
        
        num_classes = MarkupText(f'Los insectos pertencen a <span fgcolor="{BLUE}">102</span> clases', 
                               color=WHITE).scale(0.7)
        
        division = MarkupText(f'Los datos están divididos en:', color=WHITE).scale(0.7)
        division2 = MarkupText(f'<span fgcolor="{YELLOW}">45,095</span> imágenes de entrenamiento\n<span fgcolor="{TEAL}">7,508</span> imágenes de validación\ny <span fgcolor="{PURPLE}">22,619</span> imágenes de prueba').scale(0.7)
        
        text = [
            num_datos,
            num_classes,
            division]
        
        digito.scale(.2).next_to(norm, DOWN)
        vals.scale(.2).next_to(norm, DOWN)
        vals_norm.scale(.2).next_to(norm, DOWN)

        # Hacer zoom al centro de los valores en un cuadro de la image
        self.zoomed_camera.frame.set_stroke(RED)
        self.zoomed_camera.frame.set_scale(3)
        self.zoomed_camera.frame.set_z_index(3)

        text = VGroup(*text).arrange(DOWN,buff=0.3,aligned_edge = LEFT).next_to(title, DOWN).to_edge(LEFT)
        anims = [FadeIn(VGroup(Dot().next_to(t,LEFT,buff = 0.1),t),shift=RIGHT) for t in text] + [FadeIn(division2.next_to(text, DOWN, buff=0.1).to_edge())]
        
        with self.voiceover(text="titulo_datos") as tracker:
            self.play(Write(title))
            self.wait(tracker.duration-1)
        
        for i, anim in enumerate(anims):
            with self.voiceover(text=str(i)) as tracker:
                self.play(anim, run_time=1)
                self.wait(tracker.duration-1)
        #self.play(LaggedStart(*anims ,lag_ratio = 0.5))

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        with self.voiceover(text="acaros") as tracker:
            self.play(FadeIn(acaros), Write(acaros_tex))
            self.wait(tracker.duration-2)
        self.play(FadeOut(acaros), Unwrite(acaros_tex))

        with self.voiceover(text="afidos") as tracker:
            self.play(FadeIn(afidos), Write(afidos_tex))
            self.wait(tracker.duration-2)
        self.play(FadeOut(afidos), Unwrite(afidos_tex))

        with self.voiceover(text="escara") as tracker:
            self.play(FadeIn(escara), Write(escara_tex))
            self.wait(tracker.duration-2)
        self.play(FadeOut(escara), Unwrite(escara_tex))

        with self.voiceover(text="gusano") as tracker:
            self.play(FadeIn(b), Write(b_tex))
            self.wait(tracker.duration-1)

        
        with self.voiceover(text="preproces") as tracker:
            self.play(FadeOut(b[0]), b[1].animate.center(), Transform(b_tex, tam_entrada))          
            self.wait(tracker.duration-1)
        
        with self.voiceover(text="resize") as tracker:
            self.play(b[1].animate.set_height(3.5), b[1].animate.set_width(3.5))
            self.wait(tracker.duration-1)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
            
        with self.voiceover(text="escalar") as tracker:
            self.play(FadeIn(digito), FadeIn(SurroundingRectangle(digito, buff=0, color=WHITE)), Write(norm))
            self.play(ReplacementTransform(digito, vals))
            self.play(Create(self.zoomed_camera.frame))
            self.activate_zooming()
            self.play(self.get_zoomed_display_pop_out_animation())
            self.play(self.zoomed_camera.frame.animate.move_to((-4, 1, .0)))
            self.play(Transform(vals, vals_norm))


        
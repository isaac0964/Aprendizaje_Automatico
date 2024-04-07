from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService

class Presentacion(VoiceoverScene):
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))

        # Definir Imagenes ---------------------------------------
        udg = ImageMobject("imgs/udg.png").scale(0.85)
        cucei = ImageMobject("imgs/cucei.png").scale(0.5).shift(DOWN * 0.35)

        # Definir textos ------------------------------------------
        nombre = MarkupText(
            f'Ángel <span fgcolor="{BLUE}">Isaac</span> Gómez <span fgcolor="{RED}">Canales</span>', color=WHITE, font_size=50)
        nombre.next_to(udg, DOWN, buff=0.5)

        titulo1 = Text("Aumento de Datos", color=BLUE, font_size=60)
        titulo2 = MarkupText(f'Mediante <span fgcolor="{RED}">GAN</span>', color=WHITE, font_size=60)
        titulo = VGroup(titulo1, titulo2).arrange(DOWN)

        # Intro Datos y Logos ------------------------------------
        anim_cucei = [FadeIn(cucei), cucei.animate.shift(RIGHT)]
        with self.voiceover(
            text= "Hola, Buenas tardes. Mi nombre es Ángel Isaag Gómez Canales"
        ) as tracker:
            self.play(FadeIn(udg), run_time=tracker.duration/3)
            self.play(udg.animate.shift(LEFT), AnimationGroup(*anim_cucei, lag_ratio=0.2), run_time=tracker.duration/3)
            self.play(Write(nombre), run_time=tracker.duration/3)

        # Intro Titulo -------------------------------------------
        logos = Group(cucei, udg)

        anim_mov = [logos.animate.scale(0.6).to_edge(UL), nombre.animate.shift(UP)]
        with self.voiceover(
            text="A continuación, presentaré mi proyecto final de Aprendizaje Automático, titulado Aumento de Datos Mediante GAN"
        ) as tracker:
            self.play(AnimationGroup(*anim_mov, lag_ratio=0.1), run_time=tracker.duration/4)
            self.play(Transform(nombre, titulo), run_time=tracker.duration/4)
        

class Indice(VoiceoverScene):
    def construct(self):
        
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))
        # Imagenes ----------------------------------
        pi_hola = SVGMobject("imgs/hola.svg").shift(RIGHT * 3).flip(UP)

        # Texto ----------------------------------
        titulo_indice = Text("Índice:", color=YELLOW, font_size=60).to_edge(UL)
        textos_lista = ["Introducción al Problema", "Conjuntos de Datos", "¿Qué es una GAN?", "Arquitectura", "Resultados", "Conclusiones"]
        lista = BulletedList(*textos_lista).next_to(titulo_indice, DOWN, buff=0.5)
        lista.shift(RIGHT * 1.5)
        lista.set_color_by_tex("Introducción al Problema", RED)
        lista.set_color_by_tex("Conjuntos de Datos", GREEN)
        lista.set_color_by_tex("¿Qué es una GAN?", GOLD)
        lista.set_color_by_tex("Arquitectura", BLUE)
        lista.set_color_by_tex("Resultados", PURPLE)
        lista.set_color_by_tex("Conclusiones", TEAL)


        # Animar Indice -------------------------------
        self.play(FadeIn(pi_hola), run_time=0.7)
        self.play(FadeIn(titulo_indice))
        with self.voiceover(text="Ahora veremos las secciones en las que está estructurado el video.") as tracker:
            self.wait(tracker.duration)
        for line in lista:
            with self.voiceover(text=str(line)+".") as tracker:
                self.play(Write(line), run_time=1.5)
                self.wait(tracker.duration-1.5)
        

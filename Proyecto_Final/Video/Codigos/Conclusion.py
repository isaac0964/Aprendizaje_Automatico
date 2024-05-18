from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService

class Conclu(VoiceoverScene):
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))
        title = Tex("Conclusiones", color=PURPLE, font_size=60).to_corner(UL)

        c1 = "MobileNetV2 es capaz de clasificar de manera aceptable los datos, sin embargo puede mejorar"
        c2 = "La transferencia de aprendizaje permite obtener mejores resultados"
        c3 = "Trabajo a futuro: probar modelos más grandes o módulos de atención"
        textos_lista = [c1, c2, c3]
        lista = BulletedList(*textos_lista, font_size=30).next_to(title, DOWN, buff=0.5)
        lista.to_edge(LEFT)
        lista.set_color_by_tex(c1, GREEN)
        lista.set_color_by_tex(c2, WHITE)
        lista.set_color_by_tex(c3, RED)

        with self.voiceover(text= "titu_conclu") as tracker:
            self.play(FadeIn(title))
            self.wait(tracker.duration-1)

        #with self.voiceover(text="Ahora veremos las secciones en las que está estructurado el video...") as tracker:
        #   self.wait(tracker.duration)
        for line in lista:
            with self.voiceover(text=str(line)+"...") as tracker:
                self.play(Write(line))
                self.wait(tracker.duration-1.5) 

class Adioss(VoiceoverScene):
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))
        
        gracias = Text("Gracias por ver :)", color=YELLOW, font_size=70)
        pi_hola = SVGMobject("imgs/hola.svg").next_to(gracias, DOWN)

        with self.voiceover(text="adios") as tracker:
            self.add(pi_hola)
            self.play(Write(gracias))
            self.wait(7)
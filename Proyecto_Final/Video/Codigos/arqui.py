from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService

class mobile(VoiceoverScene):

    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))

        title = Text("Arquitectura", color=GREEN).to_corner(UL)

        arquior = SVGMobject("imgs/arquior.svg").scale(1.5).shift(UP)
        bottle = SVGMobject("imgs/bottlenecks.svg").scale(1.5).next_to(arquior, DOWN, buff=0.4)
        arquimod = SVGMobject("imgs/arquimod.svg").scale(1.8)

        with self.voiceover(text= "Pres_titulo") as tracker:
            self.play(Write(title))
            self.wait(tracker.duration-1)

        with self.voiceover(text= "arquior") as tracker:
            self.play(FadeIn(arquior))
            self.wait(tracker.duration-1)

        with self.voiceover(text= "cuello") as tracker:
            self.play(FadeIn(bottle))
            self.wait(tracker.duration-1)

        self.play(FadeOut(*[mob for mob in self.mobjects]))

        with self.voiceover(text= "arquimod") as tracker:
            self.play(FadeIn(arquimod))
            self.wait(tracker.duration-1)

class entreno(VoiceoverScene):
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))
        train1 = Table([
                        [("Factor de Aprendizaje"), "0.0001"],
                        ["Tamaño de Lote", "64"],
                        ["Optimizador", "Adam"],
                        ["Learning Rate Schedule", "Decaimiento exponencial\n\t\t\t  tasa = 0.96"],
                        ["Dropout", "0.3"],
                        ["Épocas", "50"],
                        ["Funcion de Costo", "Entropía Cruzada\nCategórica(CCE)"],
                        ["Tamaño de entrada", "224 x 224 x3"]],
                        col_labels=[Text("Hiperparámetro", color=RED), Text("Valor", color=RED)],
                        include_outer_lines=True).scale(0.4).shift(DOWN*0.2)
        # BLUE color to hiperparameter names
        ent = train1.get_entries_without_labels()
        train1_tex = Text("Primer\nentrenamineto", color=YELLOW).next_to(train1, LEFT).scale(0.6)
        for i in range(8):
            ent[i*2].set_color(BLUE)

        train2 = Table([
                        [("Factor de Aprendizaje"), "0.000001"],
                        ["Tamaño de Lote", "64"],
                        ["Optimizador", "Adam"],
                        ["Learning Rate Schedule", "Decaimiento exponencial\n\t\t\t  tasa = 0.96"],
                        ["Dropout", "0.3"],
                        ["Épocas", "100"],
                        ["Funcion de Costo", "CCE"],
                        ["Tamaño de entrada", "224 x 224 x3"]],
                        col_labels=[Text("Hiperparámetro", color=RED), Text("Valor", color=RED)],
                        include_outer_lines=True).scale(0.4)
        # BLUE color to hiperparameter names
        ent = train2.get_entries_without_labels()
        train2_tex = Text("Ajuste fino", color=YELLOW).next_to(train2, LEFT).scale(0.6)
        for i in range(8):
            ent[i*2].set_color(BLUE)

        train3 = Table([
                        [("Factor de Aprendizaje"), "0.00001"],
                        ["Tamaño de Lote", "64"],
                        ["Optimizador", "Adam"],
                        ["Dropout", "0.3"],
                        ["Épocas", "25"],
                        ["Funcion de Costo", "CCE"],
                        ["Tamaño de entrada", "224 x 224 x3"]],
                        col_labels=[Text("Hiperparámetro", color=RED), Text("Valor", color=RED)],
                        include_outer_lines=True).scale(0.4)
        # BLUE color to hiperparameter names
        ent = train3.get_entries_without_labels()
        train3_tex = Text("Ajuste final", color=YELLOW).next_to(train3, LEFT).scale(0.6)
        for i in range(7):
            ent[i*2].set_color(BLUE)

        title = Text("Configuración del Entrenamiento").to_edge(UL)

        with self.voiceover(text= "config_model") as tracker:
            self.play(Write(title))
            self.wait(tracker.duration-1)

        with self.voiceover(text= "config1") as tracker:
            self.play(Create(train1), Write(train1_tex))
            self.wait(tracker.duration-1)
        self.play(FadeOut(train1), FadeOut(train1_tex))

        with self.voiceover(text= "config2") as tracker:
            self.play(Create(train2), Write(train2_tex))
            self.wait(tracker.duration-1)
        self.play(FadeOut(train2), FadeOut(train2_tex))

        with self.voiceover(text= "config3") as tracker:
            self.play(Create(train3), Write(train3_tex))
            self.wait(tracker.duration-1)
        self.play(FadeOut(train3), FadeOut(train3_tex))
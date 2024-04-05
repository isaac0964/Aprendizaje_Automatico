from manim import *

class GAN(Scene):
    def construct(self):
        # Presentacion -----------------------------------------
        cucei = ImageMobject("imgs/cucei.png").scale(0.5).shift(DOWN * 0.35)
        udg = ImageMobject("imgs/udg.png").scale(0.85)
        nombre = MarkupText(
            f'Ángel <span fgcolor="{BLUE}">Isaac</span> Gómez <span fgcolor="{RED}">Canales</span>', color=WHITE, font_size=50)
        nombre.next_to(udg, DOWN, buff=0.5)
        titulo1 = Text("Aumento de Datos", color=BLUE, font_size=60)
        titulo2 = MarkupText(f'Mediante <span fgcolor="{RED}">GAN</span>', color=WHITE, font_size=60)
        titulo = VGroup(titulo1, titulo2).arrange(DOWN)

        datos = Group(cucei, udg)

        self.play(FadeIn(udg))
        anim_cucei = [FadeIn(cucei), cucei.animate.shift(RIGHT)]
        self.play(udg.animate.shift(LEFT), AnimationGroup(*anim_cucei, lag_ratio=0.2))
        self.play(Write(nombre))

        anim_mov = [datos.animate.scale(0.6).to_edge(UL), nombre.animate.shift(UP)]

        self.play(AnimationGroup(*anim_mov, lag_ratio=0.1))
        self.play(Transform(nombre, titulo))

    
        self.wait(3)
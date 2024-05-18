from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os

logs_dir = "../Codigo/Logs"

# Concatenate history of each training to plot all history
hist1 = pd.read_csv(os.path.join(logs_dir, "history_MobileNetv2"))
hist2 = pd.read_csv(os.path.join(logs_dir, "history_mobilenetv2_fine"))
hist3 = pd.read_csv(os.path.join(logs_dir, "history_mobilenetv2_fine2"))

history_all = pd.concat([hist1, hist2, hist3])
history_all["epoch"] += 1
epoch0 = pd.DataFrame({'epoch':0, 'accuracy':0, 'loss':10,
                        'val_accuracy':0, 'val_loss':10},
                                                            index =[0])
history_all = pd.concat([epoch0, history_all])  # Add epoch 0 to dataframe

acc = history_all['accuracy'].to_numpy()
val_acc = history_all['val_accuracy'].to_numpy()

loss = history_all['loss'].to_numpy()
loss[0] = 4.5
val_loss = history_all['val_loss'].to_numpy()
val_loss[0] = 4.5

epochs = history_all["epoch"].to_numpy()
class Perdidas(VoiceoverScene):
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))

        title = Text("Resultados", color=BLUE).to_edge(UL)
        perd = Text("Pérdida").scale(0.9).to_edge(UL)

        # Value tracker to update the scene with an x-value
        vt = ValueTracker()  # Starts at 0

        # Declare x and y axes
        ax = Axes(x_range=[0, 176, 25], y_range=[0, 4.5, 0.5], axis_config={"font_size": 14}).scale(0.6)
        ax.add_coordinates()

        interp1 = interp1d(epochs, loss)
        f1 = always_redraw(lambda: ax.plot(lambda x: interp1(x),
             color=RED, x_range=[0, vt.get_value()]))
        interp2 = interp1d(epochs, val_loss)
        f2 = always_redraw(lambda: ax.plot(lambda x: interp2(x),
             color=BLUE, x_range=[0, vt.get_value()]))
        
        # Add ddot to trace the function, also pointed to the Value Tracker
        f1_dot = always_redraw(lambda: Dot(point=ax.c2p(vt.get_value(), f1.underlying_function(vt.get_value())), color=RED))
        f2_dot = always_redraw(lambda: Dot(point=ax.c2p(vt.get_value(), f2.underlying_function(vt.get_value())), color=BLUE))
        
        dot = Dot(point=ax.c2p(145, 4), color=RED)
        dot_text = Text("Entrenamiento", color=RED).scale(0.4).move_to(ax.c2p(170, 4))
        dot_val = Dot(color=BLUE).next_to(dot, DOWN)
        dot_val_text = Text("Validación", color=BLUE).scale(0.4).move_to(ax.c2p(165, 3.5))
        
        # Add number of population above dot
        loss1 = always_redraw(lambda: DecimalNumber(f1.underlying_function(vt.get_value())).set_color(RED).scale(0.5).next_to(f1_dot, DOWN))
        loss2 = always_redraw(lambda: DecimalNumber(f2.underlying_function(vt.get_value())).set_color(BLUE).scale(0.5).next_to(f2_dot, UP))

        # Add axes labels
        labels = ax.get_axis_labels(Tex("Época").scale(0.5), Tex("CCE").scale(0.5))

        with self.voiceover(text= "Pres_res") as tracker:
            self.play(Write(title))
            self.wait(tracker.duration-1)
        self.play(FadeOut(title))    
        self.play(Write(perd))

        # Animate Axis being drawn
        self.play(Write(ax), Write(labels))

        # Add function and trace dot
        self.add(f1, f1_dot, loss1)
        self.add(f2, f2_dot, loss2)
        self.add(dot, dot_text, dot_val, dot_val_text)

        with self.voiceover(text= "loss_plot") as tracker:
            self.play(vt.animate.set_value(175), run_time=8)
            self.wait(tracker.duration-8)

class Acc(VoiceoverScene):
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))

        accuracy = Text("Accuracy").scale(0.9).to_edge(UL)

        # Value tracker to update the scene with an x-value
        vt = ValueTracker()  # Starts at 0

        # Declare x and y axes
        ax = Axes(x_range=[0, 176, 25], y_range=[0, 0.8, 0.1], axis_config={"font_size": 14}).scale(0.6)
        ax.add_coordinates()

        interp1 = interp1d(epochs, acc)
        f1 = always_redraw(lambda: ax.plot(lambda x: interp1(x),
             color=RED, x_range=[0, vt.get_value()]))
        interp2 = interp1d(epochs, val_acc)
        f2 = always_redraw(lambda: ax.plot(lambda x: interp2(x),
             color=BLUE, x_range=[0, vt.get_value()]))
        
        # Add ddot to trace the function, also pointed to the Value Tracker
        f1_dot = always_redraw(lambda: Dot(point=ax.c2p(vt.get_value(), f1.underlying_function(vt.get_value())), color=RED))
        f2_dot = always_redraw(lambda: Dot(point=ax.c2p(vt.get_value(), f2.underlying_function(vt.get_value())), color=BLUE))
        
        dot = Dot(point=ax.c2p(145, 0.3), color=RED)
        dot_text = Text("Entrenamiento", color=RED).scale(0.4).move_to(ax.c2p(170, 0.3))
        dot_val = Dot(color=BLUE).next_to(dot, DOWN)
        dot_val_text = Text("Validación", color=BLUE).scale(0.4).move_to(ax.c2p(165, 0.21))
        
        # Add number of population above dot
        acc1 = always_redraw(lambda: DecimalNumber(f1.underlying_function(vt.get_value())).set_color(RED).scale(0.5).next_to(f1_dot, UP))
        acc2 = always_redraw(lambda: DecimalNumber(f2.underlying_function(vt.get_value())).set_color(BLUE).scale(0.5).next_to(f2_dot, DOWN))

        # Add axes labels
        labels = ax.get_axis_labels(Tex("Época").scale(0.5), Tex("Accuracy").scale(0.5))

        with self.voiceover(text= "Pres_acc") as tracker:
            self.play(Write(accuracy))
            self.wait(tracker.duration-1)

        # Animate Axis being drawn
        self.play(Write(ax), Write(labels))

        # Add function and trace dot
        self.add(f1, f1_dot, acc1)
        self.add(f2, f2_dot, acc2)
        self.add(dot, dot_text, dot_val, dot_val_text)

        with self.voiceover(text= "acc_plot") as tracker:
            self.play(vt.animate.set_value(175), run_time=8)
            self.wait()
            self.wait(tracker.duration-8)

class Metricas(VoiceoverScene):
    def construct(self):
        self.set_speech_service(RecorderService(silence_threshold=-40.0, transcription_model=None))
        title = Text("Metricas").to_corner(UL)
        gm = Tex(r"$\text{GM} = \prod_{i=1}^C  \sqrt[C]{S_i}$", font_size=98)

        pi_think = SVGMobject("imgs/thinking.svg").to_edge(UR).flip(UP)
        pi_hurra = SVGMobject("imgs/hurra.svg")
        gradcam = ImageMobject("imgs/gradcam.png").scale(0.23)

        metricas = Table([
                        [("Accuracy"), "61"],
                        ["Precision", "59"],
                        ["Recall", "51"],
                        ["F1", "54"],
                        ["GM", "46.43"]],
                        col_labels=[Text("Métrica", color=RED), Text("Valor (%)", color=RED)],
                        include_outer_lines=True).scale(0.5)
        # BLUE color to hiperparameter names
        ent = metricas.get_entries_without_labels()

        for i in range(5):
            ent[i*2].set_color(BLUE)

        with self.voiceover(text= "titulo_metri") as tracker:
            self.play(Write(title))
            self.wait(tracker.duration-1)
        
        with self.voiceover(text= "gm") as tracker:
            self.add(pi_think)
            self.play(Write(gm))
            self.wait(tracker.duration-1)

        self.play(FadeOut(gm))

        with self.voiceover(text= "metricas") as tracker:
            self.play(Create(metricas))
            self.wait(tracker.duration-1)

        self.play(FadeOut(metricas), FadeOut(title))

        with self.voiceover(text= "gradcam") as tracker:        
            self.play(FadeIn(gradcam))
            self.wait(tracker.duration-1)

        self.remove(pi_think)
        self.play(FadeOut(gradcam))
        with self.voiceover(text= "demo") as tracker:
            self.add(pi_hurra)        
            self.wait(tracker.duration)



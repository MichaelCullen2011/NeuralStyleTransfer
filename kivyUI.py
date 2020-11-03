import kivy

from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout


class MainScreen(GridLayout, FloatLayout):
    # def __init__(self, **kwargs):
    #     super(MainScreen, self).__init__(**kwargs)
    #     self.anchor_x = 'center'
    #     self.anchor_y = 'top'
    #     self.cols = 1
    #     self.add_widget(Label(text="Neural Style Transfer", height=200, width=500))
    #     self.anchor_x = 'left'
    #     self.anchor_y = 'center'
    #     self.cols = 1
    #     self.add_widget(Button(text="Choose Image", size_hint=(0.2, 0.3)))
    #     self.add_widget(Image(source='./images/Originals/Dog.jpg', size_hint_x=60))
    #     self.add_widget(Button(text="Choose Style", size_hint_x=60))
    #     self.add_widget(Image(source='./images/Originals/Kandinsky.jpg', size_hint_x=60))
    pass


class MyApp(App):
    def build(self):
        MainScreen()


if __name__ == '__main__':
    MyApp().run()



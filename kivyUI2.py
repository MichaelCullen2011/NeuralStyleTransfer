import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ListProperty, StringProperty
from kivy.clock import Clock
import os
import threading
import NST

Builder.load_file("my.kv")


class MainScreen(Screen):
    image_source = StringProperty('./images/Originals/Images/Dog.jpg')
    style_source = StringProperty('./images/Originals/Styles/Kandinsky.jpg')
    image_name = 'Dog.jpg'
    style_name = 'Kandinsky.jpg'

    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)


class ImageScreen(Screen):
    pass


class StyleScreen(Screen):
    pass


class RootWidget(ScreenManager):
    pass


sm = ScreenManager()


class MyApp(App):
    image_source = StringProperty('./images/Originals/Images/Dog.jpg')
    style_source = StringProperty('./images/Originals/Styles/Kandinsky.jpg')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(ImageScreen(name='image'))
        sm.add_widget(StyleScreen(name='style'))
        print(MainScreen.image_name, MainScreen.style_name)

    def choose_image(self, filename):
        file = filename[75:]
        filename = './images/Originals/Images/' + filename[75:]
        MainScreen.image_source = filename
        MainScreen.image_name = file
        self.image_source = filename

    def choose_style(self, filename):
        file = filename[75:]
        filename = './images/Originals/Styles/' + filename[75:]
        MainScreen.style_source = filename
        MainScreen.style_name = file
        self.style_source = filename

    def refresh(self):
        MyApp()

    def setup_program(self):
        # CHANGE THIS SO IT CAN ACCEPT MULTIPLE INPUT FILES IN image_to_use and style_to_use!!!!!
        # Currently only works by replacing the list completely
        image_name = MainScreen.image_name[:-4]
        style_name = MainScreen.style_name[:-4]
        NST.image_to_use = [image_name]
        NST.style_to_use = [style_name]
        print("From Kivy, image to use: ", NST.image_to_use)
        print("From Kivy, style to use: ", NST.style_to_use)
        NST.running, NST.Train, NST.Display, NST.Save = True, True, True, True
        threading.Thread(target=NST.run_nst).start()
        #NST.run_nst()

    def build(self):
        return RootWidget()


if __name__ == '__main__':
    MyApp().run()



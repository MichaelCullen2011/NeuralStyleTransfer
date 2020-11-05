import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ListProperty, StringProperty
import os
# import NST

Builder.load_file("my.kv")


class MainScreen(Screen):
    image_source = StringProperty('images/Originals/Images/Dog.jpg')
    style_source = StringProperty('./images/Originals/Styles/Kandinsky.jpg')

    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        print(self.image_source)
        print(self.style_source)


class ImageScreen(Screen):
    pass


class StyleScreen(Screen):
    pass


class RootWidget(ScreenManager):
    pass


class MyApp(App):
    image_source = StringProperty('images/Originals/Images/Dog.jpg')
    style_source = StringProperty('./images/Originals/Styles/Kandinsky.jpg')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(ImageScreen(name='image'))
        sm.add_widget(StyleScreen(name='style'))
        image_source = self.image_source
        style_source = self.style_source

    def choose_image(self, filename):
        # # 'D:\Users\micha\PycharmProjects\NeuralStyleTransfer\images\Originals\Images\Teasdale1.jpg'
        filename = './images/Originals/Images/' + filename[75:]
        MainScreen.image_source = filename
        self.image_source = filename

    def choose_style(self, filename):
        filename = './images/Originals/Styles/' + filename[75:]
        MainScreen.style_source = filename
        self.style_source = filename

    def refresh(self):
        MyApp()

    def build(self):
        return RootWidget()


if __name__ == '__main__':
    MyApp().run()



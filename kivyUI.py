import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ListProperty, StringProperty
import os
#import NST

Builder.load_string("""
# Colour Palette
# 1, 0.61, 0.59, 1
# 1, 0.70, 0.66, 1
# 1, 0.78, 0.76, 1
# 0.98, 0.85, 0.76, 1
# 0.98, 0.79, 0.65, 1

<CustomLabel1@Label>:
    font_size: 25
    color: 1, 1, 1, 1

<CustomButton1@Button>:
    size_hint: 0.2, 0.2
    font_size: 20
    color: 1, 0.61, 0.59, 1
    background_color: 0.98, 0.79, 0.65, 1
    
<CustomButton2@Button>:
    size_hint: 0.1, 0.1
    font_size: 20
    color: 1, 0.61, 0.59, 1
    background_color: 0.98, 0.79, 0.65, 1
    

<MainScreen>:
    FloatLayout:
        CustomLabel1:
            id: 'title_label'
            pos_hint: {"x":0.26, "y":0.85}
            size_hint: 0.5, 0.2
            text: 'Neural Style Transfer'
        CustomButton1:
            id: 'choose_btn'
            pos_hint: {"x":0.1, "y":0.6}
            text: 'Choose Image'
            on_press: root.choose_image()
        Image:
            id: 'image_pic'
            pos_hint: {"x":0.4, "y":0.6}
            source: root.image_source
            size_hint: 0.2, 0.2
            allow_stretch: True
        CustomButton1:
            id: 'style_btn'
            pos_hint: {"x":0.1, "y":0.3}
            text: 'Choose Style'
            on_press: root.choose_style()
        Image:
            id: 'style_pic'
            pos_hint: {"x":0.4, "y":0.3}
            source: root.style_source
            size_hint: 0.2, 0.2
            allow_stretch: True

<ImageScreen>:
    text_input: ''
    FloatLayout:
        CustomLabel1:
            id: 'title_label'
            pos_hint: {"x":0.26, "y":0.85}
            size_hint: 0.5, 0.2
            text: 'Neural Style Transfer'
        FileChooserIconView:
            id: 'image_chooser'
            pos_hint: {"x":0, "y":-0.1}
            path: '/Users/micha/PycharmProjects/NeuralStyleTransfer/images/Originals/Images'
            on_selection: text_input.text = self.selection[0] or ''
            on_selection: image_pic.source = self.selection[0] or ''
        CustomButton2:
            id: 'submit_btn'
            pos_hint: {"x":0.85, "y":0.20}
            text: 'Submit'
            on_press: root.selected(text_input.text)
        CustomButton2:
            id: 'return_btn'
            pos_hint: {"x":0.85, "y":0.05}
            text: 'Return'
            on_press: root.switching_function()
        Image:
            id: image_pic
            pos_hint: {"x":0.3, "y":0.2}
            source: './images/Originals/Images/Dog.jpg'
            size_hint: 0.4, 0.4
            allow_stretch: True
        CustomLabel1:
            id: 'file_path_label'
            pos_hint: {"x":0.26, "y":0}
            size_hint: 0.5, 0.2
            text: 'File Path:'
        TextInput:
            id: text_input
            size_hint_y: None
            height: 30
            multiline: False

<StyleScreen>:
    FloatLayout:
        CustomLabel1:
            id: 'title_label'
            pos_hint: {"x":0.26, "y":0.85}
            size_hint: 0.5, 0.2
            text: 'Neural Style Transfer'
        FileChooserIconView:
            id: 'style_chooser'
            pos_hint: {"x":0, "y":-0.1}
            path: '/Users/micha/PycharmProjects/NeuralStyleTransfer/images/Originals/Styles'
            on_selection: text_input.text = self.selection[0] or ''
            on_selection: style_pic.source = self.selection[0]
        CustomButton2:
            id: 'submit_btn'
            pos_hint: {"x":0.85, "y":0.20}
            text: 'Submit'
            on_press: root.selected(text_input.text)
        CustomButton2:
            id: 'return_btn'
            pos_hint: {"x":0.85, "y":0.05}
            text: 'Return'
            on_press: root.switching_function()
        Image:
            id: style_pic
            pos_hint: {"x":0.3, "y":0.2}
            source: './images/Originals/Styles/Kandinsky.jpg'
            size_hint: 0.4, 0.4
            allow_stretch: True
        CustomLabel1:
            id: 'file_path_label'
            pos_hint: {"x":0.26, "y":0}
            size_hint: 0.5, 0.2
            text: 'File Path:'
        TextInput:
            id: text_input
            size_hint_y: None
            height: 30
            multiline: False
""")


class MainScreen(Screen):
    image_source = StringProperty('images/Originals/Images/Dog.jpg')
    style_source = StringProperty('images/Originals/Styles/Kandinsky.jpg')

    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)

    def choose_image(self):
        global sm
        sm.current = 'Image'

    def choose_style(self):
        global sm
        sm.current = 'Style'

    def main_return(self):
        global sm
        sm.current = 'Main'


class ImageScreen(Screen):
    def switching_function(*args):
        global sm
        sm.current = 'Main'

    def __init__(self, **kwargs):
        super(ImageScreen, self).__init__(**kwargs)

    def selected(self, filename):
        global sm
        MainScreen.image_source = StringProperty(filename)
        print(MainScreen.image_source)
        sm.current = 'Main'


class StyleScreen(Screen):
    def switching_function(*args):
        global sm
        sm.current = 'Main'

    def __init__(self, **kwargs):
        super(StyleScreen, self).__init__(**kwargs)

    def selected(self, filename):
        global sm
        MainScreen.style_source = StringProperty(filename)
        print(MainScreen.style_source)
        sm.current = 'Main'


sm = ScreenManager()
sm.add_widget(MainScreen(name='Main'))
sm.add_widget(ImageScreen(name='Image'))
sm.add_widget(StyleScreen(name='Style'))


class MyApp(App):
    def build(self):
        return sm


if __name__ == '__main__':
    MyApp().run()



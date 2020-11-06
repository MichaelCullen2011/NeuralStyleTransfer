from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty
from kivy.core.window import Window
from shutil import copyfile
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


class TrainScreen(Screen):
    pass


class ShowScreen(Screen):
    def __init__(self, **kwargs):
        super(ShowScreen, self).__init__(**kwargs)


class DragDropWindow(Screen):
    drop_file = StringProperty('')

    def __init__(self, **kwargs):
        super(DragDropWindow, self).__init__(**kwargs)
        Window.bind(on_dropfile=self._on_file_drop)

    def _on_file_drop(self, window, file_path):
        self.drop_file = file_path.decode("utf-8")  # convert byte to string
        self.ids.drop_pic.source = self.drop_file
        self.ids.drop_pic.reload()  # reload image
        edited_drop_file = ''
        last_indexed_slash = self.drop_file.rfind("\\")
        last_indexed_dot = self.drop_file.rfind(".")
        drop_file_dir = self.drop_file
        edited_drop_file = self.drop_file[last_indexed_slash:last_indexed_dot]
        copyfile(drop_file_dir, './images/Originals/Images/' + edited_drop_file + '.jpg')
        # print("drop_file: ", self.drop_file)
        # print("edited drop_file: ", edited_drop_file)


class RootWidget(ScreenManager):
    pass


sm = ScreenManager()


class MyApp(App):
    image_source = StringProperty('./images/Originals/Images/Dog.jpg')
    style_source = StringProperty('./images/Originals/Styles/Kandinsky.jpg')
    image_name = 'Dog.jpg'
    style_name = 'Kandinsky.jpg'
    gen_image = StringProperty('./images/Generated/{}-{}.jpg'.format(NST.image_to_use[0], NST.style_to_use[0]))
    drop_file = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(ImageScreen(name='image'))
        sm.add_widget(StyleScreen(name='style'))

    def choose_image(self, filename):
        file = filename[75:]
        filename = './images/Originals/Images/' + filename[75:]
        MainScreen.image_source = filename
        MainScreen.image_name = file
        self.image_source = filename
        self.image_name = file

    def choose_style(self, filename):
        file = filename[75:]
        filename = './images/Originals/Styles/' + filename[75:]
        MainScreen.style_source = filename
        MainScreen.style_name = file
        self.style_source = filename
        self.style_name = file

    def refresh(self):
        MyApp()

    def setup_program(self):
        # CHANGE THIS SO IT CAN ACCEPT MULTIPLE INPUT FILES IN image_to_use and style_to_use!!!!!
        # Currently only works by replacing the list completely
        image_name = MainScreen.image_name[:-4]
        style_name = MainScreen.style_name[:-4]
        NST.image_to_use = [image_name]
        NST.style_to_use = [style_name]
        # print("From Kivy, image to use: ", NST.image_to_use)
        # print("From Kivy, style to use: ", NST.style_to_use)
        NST.running, NST.Train, NST.Save = True, True, True
        nst_thread = threading.Thread(target=NST.run_nst)
        nst_thread.start()
        self.gen_image = './images/Generated/{}-{}.jpg'.format(NST.image_to_use[0], NST.style_to_use[0])
        gen_image = self.gen_image

    def build(self):
        return RootWidget()


if __name__ == '__main__':
    MyApp().run()



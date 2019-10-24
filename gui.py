import PySimpleGUI as sg
import utility
import neural_style

from pathlib import Path, PureWindowsPath
from datetime import datetime

def main():

    layout = [
            [sg.Text('Content Image', size=(15, 1)), sg.Input(key='content_image'), sg.FileBrowse(file_types=(('Images', '*.jpg'),('PNG', '*.png')))],
            [sg.Text('Model', size=(15, 1)), sg.Input(key='model'), sg.FileBrowse(file_types=(("Text Files", "*.pth"),))],
            [sg.Text('Output Directory', size=(15, 1)), sg.InputText(key='output_image'), sg.FolderBrowse()],
            [sg.Text('Output Size', size=(15, 1)), sg.InputText('2048', key='image_size')],
            [sg.Text('Selected GPU', size=(15, 1)), sg.InputText('1', key='cuda')],
            [sg.Button('Style', key='eval')],
            [sg.Text('_' * 80)],
            [sg.Text('Dataset Directory', size=(15, 1)), sg.InputText('D:/Neural Networks Training Sets/ImaginaryFashion', key='dataset'), sg.FolderBrowse()],
            [sg.Text('Style Image', size=(15, 1)), sg.Input(key='style_image'), sg.FileBrowse(file_types=(('Images', '*.jpg'),('PNG', '*.png')))],
            [sg.Text('Save Model Directory', size=(15, 1)), sg.InputText(key='save_model_dir'), sg.FolderBrowse()],
            [sg.Text('Epochs ', size=(15, 1)), sg.InputText('2', key='epochs')],
            [sg.Text('Seed ', size=(15, 1)), sg.InputText('123', key='seed')],
            [sg.Text('Batch Size ', size=(15, 1)), sg.InputText('1', key='batch_size')],
            [sg.Text('Selected Train GPU', size=(15, 1)), sg.InputText('1', key='cuda_train')],
            [sg.Button('Train', key='train')],
            [sg.Text('_' * 80)],
            [sg.Cancel(key='quit')],
             ]

    window = sg.Window('Neural Style Transfer', layout)

    while (True):

        # This is the code that reads and updates your window
        event, values = window.Read(timeout=100)

        if event == 'Exit' or event is None:
            break

        if event == 'quit':
            break

        if event == 'train':

            checkpoint_model_dir = Path('temp/models_chackpoint/').absolute()
            save_model_dir = Path(values['save_model_dir']).absolute()
            args = {
                'subcommand': 'train',
                'dataset': values['dataset'],
                'style_image': values['style_image'],
                'save_model_dir': PureWindowsPath(save_model_dir),
                'epochs': int(values['epochs']),
                'cuda': int(values['cuda_train']),

                'seed': int(values['seed']),
                'batch_size': int(values['batch_size']),
                'checkpoint_model_dir': PureWindowsPath(checkpoint_model_dir),
                'image_size': 256,
                'style_size': None,
                'content_weight': 1e5,
                'style_weight': 1e10,
                'lr': 1e-3,
                'log_interval': 500,
                'checkpoint_interval': 5000,
            }

            args = utility.dotdict(args)
            neural_style.check_paths(args)
            neural_style.train(args)

        if event == 'eval':

            # Set output filename
            single_datestring = datetime.strftime(datetime.now(), '%Y-%m-%d_%H.%M.%S')
            photo_path = '%s/%s_%s_single.png' % ( values['output_image'], 'output', single_datestring)

            # Resize Content image
            values['content_image'] = str(utility.openSingleAndCheck(values['content_image'], 'temp/content_image.jpg', int(values['image_size'])))

            args = {
                'subcommand': 'eval',
                'model': values['model'],
                'content_image': values['content_image'],
                'content_scale':  None,
                'export_onnx': None,
                'output_image': photo_path,
                'cuda': int(values['cuda']),
            }

            args = utility.dotdict(args)
            neural_style.stylize(args)

            sg.Popup('Completed', 'Style transfer completed')


    window.Close()   # Don't forget to close your window!

if __name__ == '__main__':
    main()

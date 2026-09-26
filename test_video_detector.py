import tempfile
import unittest
from pathlib import Path

from video_detector import compare_frame, load_annotations, load_presets, save_presets


def annotation(name='phantom03_0001.xml', width=1920, height=1080, box='10,20,30,40'):
    xmin, ymin, xmax, ymax = box.split(',')
    return f'''<annotation><size><width>{width}</width><height>{height}</height></size>
    <object><name>Drone</name><bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>
    <xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox></object></annotation>'''


class AnnotationLoaderTest(unittest.TestCase):
    def write_annotation(self, directory, name='phantom03_0001.xml', **kwargs):
        (directory / name).write_text(annotation(name, **kwargs))

    def test_loads_valid_annotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_annotation(directory)
            self.assertEqual(load_annotations(directory, 2, 1920, 1080), {1: [('Drone', (10, 20, 30, 40))]})

    def test_rejects_invalid_boxes_and_frame_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_annotation(directory, box='30,20,10,40')
            with self.assertRaises(ValueError):
                load_annotations(directory, 2, 1920, 1080)
            self.write_annotation(directory)
            self.write_annotation(directory, name='phantom03_0003.xml')
            with self.assertRaises(ValueError):
                load_annotations(directory, 2, 1920, 1080)

    def test_rejects_malformed_xml(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'phantom03_0001.xml').write_text('<annotation>')
            with self.assertRaises(ValueError):
                load_annotations(directory, 2, 1920, 1080)

    def test_compares_predictions_with_ground_truth(self):
        mean_iou, matched, failures = compare_frame(
            [('Drone', (0, 0, 10, 10))], ['Drone', 'Bird'], [(0, 0, 10, 10), (20, 20, 30, 30)],
        )
        self.assertEqual((mean_iou, matched, failures), (1.0, 1, 0))
        mean_iou, matched, failures = compare_frame([('Drone', (0, 0, 10, 10))], [], [])
        self.assertEqual((mean_iou, matched, failures), (0.0, 0, 1))

    def test_saves_and_loads_named_presets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'presets.yaml'
            presets = {'phantom03': {'video': '/videos/phantom03.mp4', 'annotations': '/annotations/phantom03'}}
            save_presets(presets, path)
            self.assertEqual(load_presets(path), presets)


if __name__ == '__main__':
    unittest.main()

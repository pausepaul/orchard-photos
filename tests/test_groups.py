from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

from server import groups
from server.cache import connect, thumbnail_path
from server.groups import THRESHOLD, Groups, gather, sharpness, stem
from server.scanner import Scanner
from server.semantic import MODEL, Encoder, SemanticIndex


def turned(degrees):
    """A unit vector in the plane of the first two axes; two of them have cosine similarity cos(angle between)."""
    vector = np.zeros(8, dtype='<f4')
    vector[0], vector[1] = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return vector


def checkerboard(size=64):
    return Image.fromarray((np.indices((size, size)).sum(axis=0) % 2 * 255).astype(np.uint8)).convert('RGB')


class GatherTests(unittest.TestCase):
    def test_bursts_group_and_different_shots_stay_apart(self):
        matrix = np.stack([turned(0), turned(5), turned(10), turned(60)])
        centres = gather(matrix)
        self.assertEqual(len(set(centres[:3])), 1)
        self.assertEqual(centres[3], 3)

    def test_a_slow_series_does_not_chain_into_one_group(self):
        # Each step is a near-duplicate of the next (cos 10° ≈ 0.985), but the ends are far apart.
        matrix = np.stack([turned(10 * step) for step in range(7)])
        centres = gather(matrix)
        for index, centre in enumerate(centres):
            self.assertGreaterEqual(float(matrix[index] @ matrix[centre]), THRESHOLD)
        self.assertGreater(len(set(centres)), 1)

    def test_clustered_grouping_matches_exact_grouping(self):
        rng = np.random.default_rng(7)
        centres = np.linalg.qr(rng.normal(size=(16, 16)))[0][:10]
        vectors = []
        for cluster in range(10):
            base = centres[cluster]
            # Add a burst of near-duplicates and several distinct vectors around each cluster centre.
            for noise in (0.0, 0.02, 0.03):
                v = base + rng.normal(scale=noise, size=16)
                vectors.append(v / np.linalg.norm(v))
            for _ in range(9):
                v = base + rng.normal(scale=0.25, size=16)
                vectors.append(v / np.linalg.norm(v))
        matrix = np.stack(vectors).astype('<f4')
        exact = gather(matrix)
        with patch.object(groups, 'EXACT_LIMIT', 0):
            clustered = gather(matrix)
        np.testing.assert_array_equal(clustered, exact)


class FamilyTests(unittest.TestCase):
    def test_size_and_copy_suffixes_share_a_stem_within_one_folder(self):
        for path in ('a/Salt-Crust-150x150.jpg', 'a/salt-crust-800x400.bk.jpg', 'a/salt-crust.jpg.webp', 'a/salt-crust@2x.png',
                     'a/salt-crust-scaled.jpg', 'a/salt-crust (1).jpg', 'a/salt-crust copy 2.jpg'):
            with self.subTest(path=path):
                self.assertEqual(stem(path), 'a/salt-crust')
        self.assertNotEqual(stem('b/salt-crust.jpg'), stem('a/salt-crust.jpg'))
        self.assertNotEqual(stem('a/salt-crust-2.jpg'), stem('a/salt-crust.jpg'))

    def test_named_versions_group_when_they_look_related_but_not_otherwise(self):
        crop, original, unrelated = turned(0), turned(40), turned(130)
        centres = gather(np.stack([crop, original, unrelated]), related=[[0, 1, 2]])
        self.assertEqual(centres[0], centres[1])
        self.assertEqual(centres[2], 2)
        self.assertNotEqual(*gather(np.stack([crop, original])))


class SharpnessTests(unittest.TestCase):
    def test_blur_lowers_sharpness_and_size_alone_does_not_raise_it(self):
        with tempfile.TemporaryDirectory() as directory:
            crisp, blurred, small = (Path(directory) / name for name in ('crisp.webp', 'blurred.webp', 'small.webp'))
            image = checkerboard().resize((512, 512), Image.Resampling.NEAREST)
            image.save(crisp, 'WEBP', lossless=True)
            image.filter(ImageFilter.GaussianBlur(4)).save(blurred, 'WEBP', lossless=True)
            image.resize((128, 128), Image.Resampling.BILINEAR).save(small, 'WEBP', lossless=True)
            self.assertGreater(sharpness(crisp), 5 * sharpness(blurred))
            self.assertGreater(sharpness(crisp), sharpness(small))


class GroupsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.scanner = Scanner(self.directory / 'photos', self.directory)
        self.semantic = SemanticIndex(self.directory, Encoder(self.directory / 'models'))
        self.groups = Groups(self.directory, self.semantic)

    def add(self, number, vector, image, width=32, camera='Pixel 9'):
        image_id = f'{number:064x}'
        image.save(thumbnail_path(self.directory, image_id), 'WEBP', lossless=True)
        with connect(self.scanner.directory, rows=True) as db:
            db.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?,?)', (image_id, width, width * image.height // image.width, 'image/png', 100, None, camera, None))
            db.execute('INSERT INTO semantic_embeddings VALUES (?,?,?)', (image_id, MODEL, vector.tobytes()))
        return image_id

    def stored(self):
        with connect(self.scanner.directory, rows=True) as db:
            return {row['image_id']: (row['group_id'], row['score']) for row in db.execute('SELECT * FROM image_groups')}

    def test_only_near_duplicates_are_stored_and_the_sharpest_scores_highest(self):
        blurry = self.add(1, turned(0), checkerboard().filter(ImageFilter.GaussianBlur(2)))
        crisp = self.add(2, turned(3), checkerboard())
        alone = self.add(3, turned(80), checkerboard())
        self.assertTrue(self.groups.update())
        stored = self.stored()
        self.assertEqual(set(stored), {blurry, crisp})
        self.assertEqual(stored[blurry][0], stored[crisp][0])
        self.assertGreater(stored[crisp][1], stored[blurry][1])
        self.assertNotIn(alone, stored)

    def test_equivalent_rows_keep_all_images_and_do_not_hide_different_pixels(self):
        photo = checkerboard()
        small = self.add(1, turned(0), photo, width=640, camera=None)
        large = self.add(2, turned(0), photo, width=1280, camera=None)
        distinct = self.add(3, turned(0), Image.new('RGB', photo.size, 'white'), width=640, camera=None)
        with patch.object(groups, 'EXACT_LIMIT', 2), patch.object(groups, 'sharpness', wraps=sharpness) as measure:
            self.assertTrue(self.groups.update())
        measure.assert_called_once()
        stored = self.stored()
        self.assertEqual(set(stored), {small, large})
        self.assertEqual(stored[small][0], stored[large][0])
        self.assertGreater(stored[large][1], stored[small][1])
        self.assertNotIn(distinct, stored)

    def test_weighted_centres_keep_the_order_of_expanded_neighbour_counts(self):
        matrix = np.stack([turned(0), turned(10), turned(20), turned(30)])
        weights = np.array([4, 1, 1, 3])
        expanded = np.repeat(matrix, weights, axis=0)
        starts = np.cumsum(weights) - weights
        np.testing.assert_array_equal(starts[gather(matrix, weights=weights)], gather(expanded)[starts])

    def test_regrouping_only_when_embeddings_change(self):
        self.add(1, turned(0), checkerboard())
        self.add(2, turned(3), checkerboard())
        self.assertTrue(self.groups.update())
        self.assertFalse(self.groups.update())
        self.add(3, turned(90), checkerboard())
        self.assertFalse(self.groups.update(), 'a new image with no near-duplicates leaves every group as it was')
        self.add(4, turned(1), checkerboard())
        self.assertTrue(self.groups.update())
        self.assertEqual(len(self.stored()), 3)

    def test_completed_groups_are_reused_on_reopen(self):
        self.add(1, turned(0), checkerboard())
        self.add(2, turned(3), checkerboard())
        self.assertTrue(self.groups.update())
        expected = self.stored()
        reopened = Groups(self.directory, self.semantic)
        with patch.object(self.semantic, 'embeddings', side_effect=AssertionError('read embeddings')):
            self.assertFalse(reopened.update())
        self.assertEqual(self.stored(), expected)

    def test_existing_groups_without_a_completion_marker_are_checked(self):
        image_id = self.add(1, turned(0), checkerboard())
        with connect(self.directory) as db:
            db.execute('INSERT INTO image_groups VALUES (?,?,?)', (image_id, image_id, 1))
        self.assertTrue(self.groups.update())
        self.assertFalse(self.stored())

    def test_renaming_a_file_can_change_groups_without_new_embeddings(self):
        original = self.add(1, turned(0), checkerboard())
        crop = self.add(2, turned(40), checkerboard())
        with connect(self.directory) as db:
            for name, image_id in [('photo.jpg', original), ('other.jpg', crop)]:
                db.execute("INSERT INTO files VALUES (?, '', ?, 100, 0, '', NULL)", (name, image_id))
        self.assertFalse(self.groups.update())
        with connect(self.directory) as db:
            db.execute("UPDATE files SET path = 'photo copy.jpg' WHERE image_id = ?", (crop,))
        self.assertTrue(self.groups.update())
        self.assertEqual(self.stored()[original][0], self.stored()[crop][0])

    def test_updating_a_vector_in_place_invalidates_the_cached_matrix(self):
        self.add(1, turned(0), checkerboard())
        changed = self.add(2, turned(3), checkerboard())
        self.assertTrue(self.groups.update())
        with connect(self.directory) as db:
            db.execute("UPDATE semantic_embeddings SET vector = ? WHERE image_id = ?",
                       (turned(90).tobytes(), changed))
        self.assertTrue(self.groups.update())
        self.assertFalse(self.stored())

    def test_input_change_during_grouping_invalidates_the_result(self):
        self.add(1, turned(0), checkerboard())
        newer = self.add(2, turned(3), checkerboard())

        def gather_and_change(*args):
            centres = gather(*args)
            with connect(self.directory) as db:
                db.execute("INSERT OR REPLACE INTO semantic_embeddings VALUES (?,?,?)",
                           (newer, MODEL, turned(90).tobytes()))
            return centres

        with patch.object(groups, 'gather', side_effect=gather_and_change):
            self.assertFalse(self.groups.update())
        self.assertFalse(self.stored())
        with patch.object(groups, 'gather', wraps=gather) as gather_again:
            self.assertFalse(self.groups.update())
            gather_again.assert_called_once()
        self.assertFalse(self.stored())

    def test_the_larger_of_two_equally_sharp_copies_scores_higher(self):
        small = self.add(1, turned(0), checkerboard(), width=600)
        large = self.add(2, turned(1), checkerboard(), width=4000)
        self.groups.update()
        stored = self.stored()
        self.assertGreater(stored[large][1], stored[small][1])

    def test_screens_without_camera_details_stack_only_when_their_pixels_match(self):
        screen = Image.new('RGB', (320, 200), 'white')
        other = screen.copy()
        other.paste('black', (40, 40, 280, 160))
        shot = self.add(1, turned(0), screen, width=1440, camera=None)
        copy = self.add(2, turned(1), screen.resize((160, 100)), width=720, camera=None)
        different = self.add(3, turned(2), other, width=1440, camera=None)
        self.groups.update()
        stored = self.stored()
        self.assertEqual(stored[shot][0], stored[copy][0])
        self.assertNotIn(different, stored)

    def test_a_crop_without_camera_details_is_left_to_its_vector(self):
        photo = checkerboard(128)
        whole = self.add(1, turned(0), photo, width=1280, camera=None)
        crop = self.add(2, turned(1), photo.crop((0, 0, 128, 64)), width=1280, camera=None)
        self.groups.update()
        self.assertEqual(self.stored()[whole][0], self.stored()[crop][0])


if __name__ == '__main__':
    unittest.main()

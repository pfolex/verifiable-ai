import unittest
from aix_zero.experiment_splits import build_split_manifest, eligible_training_rows, verify_manifest


def identity(row):
    return row['prompt'].strip().lower()


class SplitTests(unittest.TestCase):
    def splits(self):
        return {s: [dict(prompt=s, answer='1', family=s)] for s in ('train', 'development', 'regression', 'final')}

    def test_duplicate_and_reserved_family_rejected(self):
        rows = self.splits(); rows['development'][0]['prompt'] = ' TRAIN '
        with self.assertRaises(ValueError): build_split_manifest(rows, identity)
        rows = self.splits(); rows['final'][0]['family'] = 'train'
        with self.assertRaises(ValueError):
            build_split_manifest(rows, identity, family_identity=lambda r:r['family'])

    def test_manifest_integrity_ingestion_and_copy_isolation(self):
        rows = self.splits(); manifest = build_split_manifest(rows, identity)
        self.assertEqual(manifest, build_split_manifest(rows, identity))
        result = eligible_training_rows(manifest, rows['train'], identity)
        result[0]['answer'] = 'changed'
        self.assertEqual(verify_manifest(manifest)['splits']['train'][0]['row']['answer'], '1')
        for split in ('development', 'regression', 'final'):
            with self.assertRaises(ValueError): eligible_training_rows(manifest, rows[split], identity)
        with self.assertRaises(ValueError):
            eligible_training_rows(manifest, [dict(rows['train'][0], answer='changed')], identity)
        manifest['payload']['splits']['final'][0]['row']['answer'] = 'changed'
        with self.assertRaises(ValueError): verify_manifest(manifest)

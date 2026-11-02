import hashlib
import tempfile
import unittest
from pathlib import Path

import scripts.cloud_preflight as cloud_preflight


class CloudPreflightTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        classifier = root / 'classifier'
        response = root / 'response'
        adapter = root / 'adapter'
        profile = root / 'profile'
        for path in (classifier, response, adapter, profile):
            path.mkdir()
        (classifier / 'model.safetensors').write_bytes(b'classifier')
        (response / 'model.safetensors').write_bytes(b'base')
        (adapter / 'adapter_model.safetensors').write_bytes(b'adapter')
        (profile / 'heads.safetensors').write_bytes(b'profile-head')
        config = {
            'classifier_dir': classifier,
            'response_model_dir': response,
            'response_adapter_dir': adapter,
            'profile_mode': 'independent',
            'profile_model_dir': profile,
        }
        manifest = {
            'classifier_files': [{
                'path': 'model.safetensors',
                'sha256': hashlib.sha256(b'classifier').hexdigest(),
            }],
            'response_adapter_sha256': hashlib.sha256(b'adapter').hexdigest(),
            'response_base_weights_sha256': hashlib.sha256(b'base').hexdigest(),
            'profile_checkpoint_tree_sha256': 'b8c7379bc998718711910d24cbb89b04fa3ccfdc4bc07ddaecb8fcc5fe42fc67',
        }
        return config, manifest, profile

    def test_independent_profile_tree_hash_is_verified(self):
        verify = getattr(cloud_preflight, 'verify_model_hashes', None)
        self.assertTrue(callable(verify), 'cloud_preflight must expose model hash verification')
        config, manifest, _ = self.fixture()
        verify(config, manifest)

    def test_independent_profile_tree_hash_mismatch_is_rejected(self):
        verify = getattr(cloud_preflight, 'verify_model_hashes', None)
        self.assertTrue(callable(verify), 'cloud_preflight must expose model hash verification')
        config, manifest, profile = self.fixture()
        (profile / 'heads.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'profile checkpoint tree hash mismatch'):
            verify(config, manifest)

    def test_empty_profile_mode_does_not_require_profile_tree_hash(self):
        verify = getattr(cloud_preflight, 'verify_model_hashes', None)
        self.assertTrue(callable(verify), 'cloud_preflight must expose model hash verification')
        config, manifest, _ = self.fixture()
        config['profile_mode'] = 'empty'
        config.pop('profile_model_dir')
        manifest.pop('profile_checkpoint_tree_sha256')
        verify(config, manifest)

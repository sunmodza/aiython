import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiython._subprocess_worker import _run


class SubprocessWorkerEdgeTests(unittest.TestCase):
    def test_worker_exception_is_serialized_for_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / 'request.pickle'
            request.write_bytes(pickle.dumps((None, 'worker', 'answer', (), {})))
            with patch('aiython.collaboration.worker_entry', side_effect=ValueError('worker failed')):
                status, kind, message = pickle.loads(_run(request))
        self.assertEqual((status, kind, message), ('error', 'ValueError', 'worker failed'))


if __name__ == '__main__':
    unittest.main()

# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import tempfile
import unittest
from professor.utils import last_checkpoint


def touch(path):
    open(path, "w").close()


class TestLastCheckpoint(unittest.TestCase):

    def test_missing_or_empty_directory(self):
        with tempfile.TemporaryDirectory() as run:
            self.assertEqual(last_checkpoint(run), ("", -1))
            self.assertEqual(last_checkpoint(os.path.join(run, "missing")), ("", -1))

    def test_bare_names(self):
        # runs written between the 3D generators and the return of the suffix use <run>/NNNN
        with tempfile.TemporaryDirectory() as run:
            for name in ("0000", "0005", "0010", "args.txt", "model.info", "vis.yaml"):
                touch(os.path.join(run, name))
            os.mkdir(os.path.join(run, "0099"))  # not a checkpoint
            self.assertEqual(last_checkpoint(run), (os.path.join(run, "0010"), 10))

    def test_pt_names_and_trailing_slash(self):
        # prof-trainer saves <run>/NNNN.pt
        with tempfile.TemporaryDirectory() as run:
            for name in ("0000.pt", "0024.pt", "events.out.tfevents.1.host"):
                touch(os.path.join(run, name))
            path, epoch = last_checkpoint(run + "/")
            self.assertEqual(epoch, 24)
            self.assertEqual(os.path.realpath(path), os.path.realpath(os.path.join(run, "0024.pt")))

    def test_mixed_names_past_9999(self):
        # the highest epoch wins, numerically
        with tempfile.TemporaryDirectory() as run:
            for name in ("0099.pt", "9999", "10000"):
                touch(os.path.join(run, name))
            self.assertEqual(last_checkpoint(run), (os.path.join(run, "10000"), 10000))


if __name__ == "__main__":
    unittest.main()

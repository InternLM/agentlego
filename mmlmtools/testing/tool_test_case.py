# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from unittest import TestCase as BaseTestCase


class ToolTestCase(BaseTestCase):

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        return self.tempdir.cleanup()

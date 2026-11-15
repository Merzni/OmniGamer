import importlib
import pkgutil
import unittest

import omnigamer


class ImportSmokeTests(unittest.TestCase):
    def test_all_application_modules_import_without_starting_services(self):
        modules = [item.name for item in pkgutil.walk_packages(omnigamer.__path__, "omnigamer.")]
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_public_tools_modules_import_without_external_detector_runtime(self):
        for module in ("tools.mark_regions", "tools.calibrate_region_metric", "tools.summarize_run"):
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()

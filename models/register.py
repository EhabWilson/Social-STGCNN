# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module register."""

import importlib
import os
import sys
from absl import logging


class Register:
    """Module register"""

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." %
                            (key, self._name))
        self._dict[key] = value

    def register(self, param):
        """Decorator to register a function or class."""

        def decorator(key, value):
            self[key] = value
            return value

        if callable(param):
            # @reg.register
            return decorator(None, param)
        # @reg.register('alias')
        return lambda x: decorator(param, x)

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            logging.error(f"module {key} not found: {e}")
            raise e

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()


class Registers():  # pylint: disable=invalid-name, too-few-public-methods
    """All module registers."""

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    model = Register('model')


MODELS = [
    "social_stgcnn",
    "social_stgcnn1",
    "social_stgcnn2",
    "social_stgcnn3",
    "social_stgcnn4",
    "social_stgcnn5",
    "social_stgcnn6",
    "social_stgcnn7",
    "social_stgcnn8",
]

ALL_MODELS = [
    ("models", MODELS)
]


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logging.warning("Module {} import failed: {}".format(name, err))
    logging.fatal("Please check these modules.")


def path_to_module_format(py_path):
    """Transform a python file path to module format."""
    return py_path.replace("/", ".").rstrip(".py")


def add_custom_modules(all_modules, config=None):
    """Add custom modules to all_modules"""
    current_work_dir = os.getcwd()
    if current_work_dir not in sys.path:
        sys.path.append(current_work_dir)
    if config is not None and "custom_modules" in config:
        custom_modules = config["custom_modules"]
        if not isinstance(custom_modules, list):
            custom_modules = [custom_modules]
        all_modules += [("", [path_to_module_format(module)])
                        for module in custom_modules]


def import_all_modules_for_register(config=None, only_nlp=False):
    """Import all modules for register."""
    all_modules = ALL_MODELS

    add_custom_modules(all_modules, config)

    logging.debug(f"All modules: {all_modules}")
    errors = []
    print(sys.path)
    for base_dir, modules in all_modules:
        for name in modules:
            try:
                if base_dir != "":
                    full_name = base_dir + "." + name
                else:
                    full_name = name
                importlib.import_module(full_name)
                logging.debug(f"{full_name} loaded.")
            except ImportError as error:
                errors.append((name, error))
    _handle_errors(errors)

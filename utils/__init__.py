"""RandOpt utilities package.

This file makes `utils` a regular package (rather than a PEP-420 namespace
package). vLLM resolves --worker-extension-cls via importlib.import_module in
spawned worker / engine-core processes (run through runpy as __mp_main__), where
namespace-package __path__ resolution is unreliable and `import utils.worker_extn`
can fail even with the directory on sys.path. A regular package imports
deterministically from the first matching sys.path entry.
"""
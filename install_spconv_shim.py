"""Register spconv shim in sys.modules before any Pointcept import."""
import sys
import types
import spconv_shim

# Create a proper module hierarchy
spconv_module = types.ModuleType('spconv')
spconv_module.pytorch = spconv_shim
spconv_module.SparseConvTensor = spconv_shim.SparseConvTensor
spconv_module.SubMConv3d = spconv_shim.SubMConv3d
spconv_module.modules = spconv_shim.modules

# Register in sys.modules
sys.modules['spconv'] = spconv_module
sys.modules['spconv.pytorch'] = spconv_shim

import moderngl

__version__ = "0.0.1"

# initiate a global opengl context for usage in ipython notebooks and so on
ctx = moderngl.create_standalone_context()  # global OpenGL context

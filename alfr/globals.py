import moderngl

__version__ = "0.0.1"

# initiate a global opengl context for usage in ipython notebooks and so on
# ctx = moderngl.create_standalone_context()  # global OpenGL context


import moderngl


class ContextManager:
    """ContextManager for OpenGL contexts with moderngl.
    based on: https://github.com/moderngl/moderngl/blob/master/examples/context_manager.py
    """

    ctx = None

    @staticmethod
    def get_default_context(allow_fallback_egl_context=True) -> moderngl.Context:
        """
        Default context
        """

        if ContextManager.ctx is None:
            try:
                ContextManager.ctx = moderngl.create_standalone_context()
            except:
                if allow_fallback_egl_context:
                    ContextManager.ctx = moderngl.create_standalone_context(
                        backend="egl"
                    )
                else:
                    raise

        return ContextManager.ctx

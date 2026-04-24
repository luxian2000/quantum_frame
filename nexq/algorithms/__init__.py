"""nexq.algorithms

高层量子算法包骨架。该包包含若干子模块（universal、qml、variational、qas、chemistry、optimizers），
用于放置通用算法、变分框架、量子化学工具、优化器等。当前仅创建模块骨架，未包含具体实现。

示例:

    from nexq import algorithms
    from nexq.algorithms import universal

"""

from . import universal, qml, variational, qas, chemistry, optimizers

__all__ = ["universal", "qml", "variational", "qas", "chemistry", "optimizers"]

from .processor_t2i import do_train as do_train_t2i
from .processor_i2i import do_train as do_train_i2i
from .processor_t2i import do_inference as do_inference_t2i
from .processor_i2i import do_inference as do_inference_i2i

__all__ = ['do_train_t2i', 'do_train_i2i', 'do_inference_t2i', 'do_inference_i2i']  

def expose_dp():
    import sys
    import types
    import Processor.DeepPhonemizer.dp as dp

    deep_phonemizer = types.ModuleType("DeepPhonemizer")
    deep_phonemizer.dp = dp

    sys.modules["DeepPhonemizer"] = deep_phonemizer
    sys.modules["DeepPhonemizer.dp"] = dp